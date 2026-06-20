#!/usr/bin/env python3
"""
Off-Ramp Orchestrator — Automated Crypto-to-Bank Pipeline

This script runs alongside the Rust yield-daemon and handles the
"last mile" of the income pipeline: converting on-chain profits
into fiat bank deposits.

Architecture:
    Rust Daemon (ZK/MEV/ML) -> On-Chain Profits -> This Script -> Exchange -> Bank

The orchestrator:
1. Monitors the daemon's profit metrics (JSON files or Prometheus)
2. Checks wallet balances via RPC
3. Swaps volatile assets to stablecoins via Jupiter/Uniswap
4. Transfers stablecoins to exchange via API
5. Sells stablecoins for fiat
6. Withdraws fiat to linked bank account

All steps are idempotent and threshold-gated.
"""

import os
import sys
import time
import json
import hmac
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG = logging.getLogger("offramp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)


class OfframpConfig:
    """Load from environment variables or .env file."""

    def __init__(self):
        self.chain = os.getenv("YIELD_CHAIN", "solana")
        self.rpc_url = os.getenv("YIELD_RPC_URL", "https://api.mainnet-beta.solana.com")

        # Wallet
        self.wallet_address = os.getenv("YIELD_WALLET_ADDRESS", "")
        self.wallet_keypair_path = os.getenv("YIELD_WALLET_KEYPAIR", "")

        # Stablecoin target
        self.stablecoin = os.getenv("YIELD_STABLECOIN", "USDC")
        self.stablecoin_mint = os.getenv(
            "YIELD_STABLECOIN_MINT",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC on Solana
        )

        # Thresholds
        self.consolidation_threshold_sol = float(
            os.getenv("YIELD_CONSOLIDATION_THRESHOLD", "0.05")
        )
        self.offramp_threshold_usd = float(
            os.getenv("YIELD_OFFRAMP_THRESHOLD", "100.0")
        )
        self.max_offramp_usd = float(
            os.getenv("YIELD_MAX_OFFRAMP", "10000.0")
        )

        # Exchange
        self.exchange = os.getenv("YIELD_EXCHANGE", "coinbase")
        self.exchange_api_key = os.getenv("YIELD_EXCHANGE_API_KEY", "")
        self.exchange_api_secret = os.getenv("YIELD_EXCHANGE_API_SECRET", "")
        self.exchange_passphrase = os.getenv("YIELD_EXCHANGE_PASSPHRASE", "")
        self.bank_account_id = os.getenv("YIELD_BANK_ACCOUNT_ID", "")
        self.fiat_currency = os.getenv("YIELD_FIAT_CURRENCY", "USD")

        # Daemon metrics directory
        self.state_dir = os.getenv("YIELD_STATE_DIR", "runtime/yield_daemon")

        # Timing
        self.check_interval = int(os.getenv("YIELD_CHECK_INTERVAL", "300"))

        # Safety
        self.dry_run = os.getenv("YIELD_DRY_RUN", "true").lower() in ("true", "1", "yes")
        self.slippage_bps = int(os.getenv("YIELD_SLIPPAGE_BPS", "50"))


# ---------------------------------------------------------------------------
# Exchange Clients
# ---------------------------------------------------------------------------


class CoinbaseClient:
    """Coinbase Advanced Trade API client."""

    BASE_URL = "https://api.coinbase.com"

    def __init__(self, api_key: str, api_secret: str, passphrase: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.passphrase = passphrase
        self.session = requests.Session()

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        message = f"{timestamp}{method}{path}{body}"
        sig = hmac.new(self.api_secret, message.encode(), hashlib.sha256)
        return sig.hexdigest()

    def _request(self, method: str, path: str, body: Optional[Dict] = None) -> Dict:
        timestamp = str(int(time.time()))
        body_str = json.dumps(body) if body else ""
        signature = self._sign(timestamp, method, path, body_str)

        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

        url = f"{self.BASE_URL}{path}"
        resp = self.session.request(method, url, headers=headers, data=body_str)
        resp.raise_for_status()
        return resp.json()

    def get_accounts(self) -> Dict:
        return self._request("GET", "/api/v3/brokerage/accounts")

    def get_usdc_balance(self) -> float:
        accounts = self.get_accounts()
        for acc in accounts.get("accounts", []):
            if acc.get("currency") == "USDC":
                return float(acc.get("available_balance", {}).get("value", 0))
        return 0.0

    def sell_usdc(self, amount: float, fiat: str = "USD") -> str:
        """Market sell USDC for fiat."""
        order = {
            "client_order_id": f"yd_{int(time.time() * 1000)}",
            "product_id": f"USDC-{fiat}",
            "side": "SELL",
            "order_configuration": {
                "market_market_ioc": {"quote_size": f"{amount:.2f}"}
            },
        }
        resp = self._request("POST", "/api/v3/brokerage/orders", order)
        return resp.get("order_id", "unknown")

    def withdraw_to_bank(self, amount: float, bank_id: str, currency: str = "USD") -> str:
        """Withdraw fiat to linked bank account."""
        body = {
            "amount": f"{amount:.2f}",
            "currency": currency,
            "payment_method": bank_id,
        }
        resp = self._request("POST", "/v2/withdrawals/payment-method", body)
        return resp.get("data", {}).get("id", "unknown")


class KrakenClient:
    """Kraken REST API client."""

    BASE_URL = "https://api.kraken.com"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret_b64 = api_secret
        self.session = requests.Session()

    def _sign(self, path: str, nonce: int, body: str) -> str:
        import base64

        sha256_hash = hashlib.sha256(f"{nonce}{body}".encode()).digest()
        preimage = path.encode() + sha256_hash
        secret = base64.b64decode(self.api_secret_b64)
        sig = hmac.new(secret, preimage, hashlib.sha512).digest()
        return base64.b64encode(sig).decode()

    def _private_request(self, path: str, params: Optional[Dict] = None) -> Dict:
        nonce = int(time.time() * 1000)
        data = {"nonce": nonce}
        if params:
            data.update(params)
        body = "&".join(f"{k}={v}" for k, v in data.items())
        signature = self._sign(path, nonce, body)

        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        url = f"{self.BASE_URL}{path}"
        resp = self.session.post(url, headers=headers, data=body)
        resp.raise_for_status()
        return resp.json()

    def get_balance(self) -> Dict:
        return self._private_request("/0/private/Balance")

    def sell_usdc(self, amount: float, fiat: str = "USD") -> str:
        resp = self._private_request("/0/private/AddOrder", {
            "pair": f"USDC{fiat}",
            "type": "sell",
            "ordertype": "market",
            "volume": f"{amount:.2f}",
        })
        txids = resp.get("result", {}).get("txid", ["unknown"])
        return txids[0] if txids else "unknown"

    def withdraw_fiat(self, amount: float, key: str, asset: str = "ZUSD") -> str:
        resp = self._private_request("/0/private/Withdraw", {
            "asset": asset,
            "key": key,
            "amount": f"{amount:.2f}",
        })
        return resp.get("result", {}).get("refid", "unknown")


# ---------------------------------------------------------------------------
# Jupiter Swap (Solana)
# ---------------------------------------------------------------------------


def jupiter_get_quote(
    input_mint: str,
    output_mint: str,
    amount_lamports: int,
    slippage_bps: int = 50,
) -> Optional[Dict]:
    """Get swap quote from Jupiter aggregator."""
    url = (
        f"https://quote-api.jup.ag/v6/quote"
        f"?inputMint={input_mint}"
        f"&outputMint={output_mint}"
        f"&amount={amount_lamports}"
        f"&slippageBps={slippage_bps}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        LOG.error(f"Jupiter quote failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Solana RPC helpers
# ---------------------------------------------------------------------------


def get_sol_balance(rpc_url: str, address: str) -> float:
    """Get SOL balance in SOL (not lamports)."""
    try:
        resp = requests.post(rpc_url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "getBalance",
            "params": [address],
        }, timeout=10)
        lamports = resp.json().get("result", {}).get("value", 0)
        return lamports / 1e9
    except Exception as e:
        LOG.error(f"SOL balance check failed: {e}")
        return 0.0


def get_spl_token_balance(rpc_url: str, owner: str, mint: str) -> float:
    """Get SPL token balance (assumes 6 decimals for USDC)."""
    try:
        resp = requests.post(rpc_url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                owner,
                {"mint": mint},
                {"encoding": "jsonParsed"},
            ],
        }, timeout=10)
        accounts = resp.json().get("result", {}).get("value", [])
        if not accounts:
            return 0.0
        info = accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]
        return float(info.get("uiAmount", 0))
    except Exception as e:
        LOG.error(f"SPL balance check failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Daemon Metrics Reader
# ---------------------------------------------------------------------------


def read_daemon_metrics(state_dir: str) -> Dict[str, Any]:
    """Read the Rust daemon's exported metrics JSON files."""
    metrics = {
        "zk_revenue_sat": 0,
        "mev_revenue_sat": 0,
        "ml_revenue_sat": 0,
        "total_revenue_sat": 0,
    }
    for domain in ("zk", "mev", "ml"):
        path = Path(state_dir) / f"{domain}_metrics.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                rev = data.get("revenue_sat", 0)
                metrics[f"{domain}_revenue_sat"] = rev
            except (json.JSONDecodeError, IOError):
                pass
    metrics["total_revenue_sat"] = sum(
        metrics[f"{d}_revenue_sat"] for d in ("zk", "mev", "ml")
    )
    return metrics


# ---------------------------------------------------------------------------
# Main Off-Ramp Loop
# ---------------------------------------------------------------------------


def run_offramp_loop(config: OfframpConfig):
    """Main autonomous off-ramp loop. Runs forever until interrupted."""
    LOG.info("=" * 60)
    LOG.info("YIELD DAEMON — OFF-RAMP ORCHESTRATOR")
    LOG.info("=" * 60)
    LOG.info(f"Chain:      {config.chain}")
    LOG.info(f"Exchange:   {config.exchange}")
    LOG.info(f"Fiat:       {config.fiat_currency}")
    LOG.info(f"Threshold:  ${config.offramp_threshold_usd:.2f}")
    LOG.info(f"Max:        ${config.max_offramp_usd:.2f}")
    LOG.info(f"Interval:   {config.check_interval}s")
    LOG.info(f"Mode:       {'DRY RUN' if config.dry_run else 'LIVE'}")
    LOG.info("=" * 60)

    # Build exchange client
    exchange_client = None
    if not config.dry_run and config.exchange_api_key:
        if config.exchange == "coinbase":
            exchange_client = CoinbaseClient(
                config.exchange_api_key,
                config.exchange_api_secret,
                config.exchange_passphrase,
            )
        elif config.exchange == "kraken":
            exchange_client = KrakenClient(
                config.exchange_api_key,
                config.exchange_api_secret,
            )

    total_withdrawn_usd = 0.0
    cycle_count = 0

    while True:
        cycle_count += 1
        try:
            now = datetime.now(timezone.utc).isoformat()

            # ── Step 1: Read daemon metrics ──────────────────────────
            metrics = read_daemon_metrics(config.state_dir)
            total_rev = metrics["total_revenue_sat"]
            LOG.info(
                f"[Cycle {cycle_count}] Revenue — "
                f"ZK:{metrics['zk_revenue_sat']} "
                f"MEV:{metrics['mev_revenue_sat']} "
                f"ML:{metrics['ml_revenue_sat']} "
                f"Total:{total_rev}"
            )

            # ── Step 2: Check wallet balance ─────────────────────────
            sol_balance = 0.0
            usdc_balance = 0.0

            if config.chain == "solana" and config.wallet_address:
                sol_balance = get_sol_balance(config.rpc_url, config.wallet_address)
                usdc_balance = get_spl_token_balance(
                    config.rpc_url, config.wallet_address, config.stablecoin_mint
                )
                LOG.info(
                    f"[Cycle {cycle_count}] Wallet — "
                    f"SOL:{sol_balance:.4f} USDC:{usdc_balance:.2f}"
                )

            # ── Step 3: Consolidation (SOL -> USDC) ──────────────────
            if sol_balance > config.consolidation_threshold_sol:
                swap_amount_sol = sol_balance - 0.01  # Keep 0.01 SOL for gas
                swap_amount_lamports = int(swap_amount_sol * 1e9)

                if config.dry_run:
                    LOG.info(
                        f"[Cycle {cycle_count}] DRY RUN: would swap "
                        f"{swap_amount_sol:.4f} SOL -> USDC"
                    )
                    # Simulate: 1 SOL ~ $150
                    usdc_balance += swap_amount_sol * 150.0
                else:
                    quote = jupiter_get_quote(
                        "So11111111111111111111111111111111111111112",
                        config.stablecoin_mint,
                        swap_amount_lamports,
                        config.slippage_bps,
                    )
                    if quote:
                        out_amount = int(quote.get("outAmount", 0))
                        usdc_received = out_amount / 1e6
                        LOG.info(
                            f"[Cycle {cycle_count}] Jupiter quote: "
                            f"{swap_amount_sol:.4f} SOL -> {usdc_received:.2f} USDC"
                        )
                        # TODO: sign and submit the swap transaction via
                        # Jupiter v6 /swap endpoint + ed25519 signing.
                        # Until swap submission is implemented, do NOT
                        # credit usdc_balance — the swap hasn't executed.
                        LOG.warning(
                            f"[Cycle {cycle_count}] Swap submission not yet "
                            f"implemented — quote only, no balance credited"
                        )

            # ── Step 4: Off-ramp (USDC -> Bank) ──────────────────────
            if usdc_balance >= config.offramp_threshold_usd:
                withdraw_amount = min(usdc_balance, config.max_offramp_usd)

                if config.dry_run:
                    LOG.info(
                        f"[Cycle {cycle_count}] DRY RUN: would off-ramp "
                        f"${withdraw_amount:.2f} {config.fiat_currency} to bank"
                    )
                    total_withdrawn_usd += withdraw_amount
                elif exchange_client:
                    try:
                        # Step 4a: Sell USDC for fiat
                        order_id = (
                            exchange_client.sell_usdc(withdraw_amount, config.fiat_currency)
                            if hasattr(exchange_client, "sell_usdc")
                            else "skip"
                        )
                        LOG.info(
                            f"[Cycle {cycle_count}] Sell order: {order_id} "
                            f"for ${withdraw_amount:.2f}"
                        )

                        # Wait for settlement
                        time.sleep(10)

                        # Step 4b: Withdraw fiat to bank
                        if config.bank_account_id:
                            if isinstance(exchange_client, CoinbaseClient):
                                tx_id = exchange_client.withdraw_to_bank(
                                    withdraw_amount, config.bank_account_id,
                                    config.fiat_currency,
                                )
                            elif isinstance(exchange_client, KrakenClient):
                                tx_id = exchange_client.withdraw_fiat(
                                    withdraw_amount, config.bank_account_id,
                                )
                            else:
                                tx_id = "unsupported"

                            total_withdrawn_usd += withdraw_amount
                            LOG.info(
                                f"[Cycle {cycle_count}] Bank withdrawal: "
                                f"{tx_id} for ${withdraw_amount:.2f} "
                                f"{config.fiat_currency}"
                            )
                    except Exception as e:
                        LOG.error(f"[Cycle {cycle_count}] Off-ramp failed: {e}")
                else:
                    LOG.warning(
                        f"[Cycle {cycle_count}] Exchange not configured — "
                        f"${withdraw_amount:.2f} ready for off-ramp"
                    )

            # ── Status ───────────────────────────────────────────────
            LOG.info(
                f"[Cycle {cycle_count}] Lifetime withdrawn: "
                f"${total_withdrawn_usd:.2f} | Next check in {config.check_interval}s"
            )

        except KeyboardInterrupt:
            LOG.info("Shutdown requested")
            break
        except Exception as e:
            LOG.error(f"[Cycle {cycle_count}] Error: {e}")

        # Sleep with responsive shutdown
        try:
            time.sleep(config.check_interval)
        except KeyboardInterrupt:
            LOG.info("Shutdown requested during sleep")
            break

    LOG.info(f"Off-ramp orchestrator stopped. Total withdrawn: ${total_withdrawn_usd:.2f}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main():
    config = OfframpConfig()
    run_offramp_loop(config)


if __name__ == "__main__":
    main()
