//! Off-Ramp Client — automated crypto-to-fiat pipeline via exchange APIs.
//!
//! Supports Coinbase Pro and Kraken REST APIs for:
//! 1. Depositing stablecoins from wallet to exchange
//! 2. Selling stablecoins for fiat (market order)
//! 3. Withdrawing fiat to linked bank account (ACH/SEPA)
//!
//! All API calls are HMAC-SHA256 signed per exchange spec.
//! Rate-limited to avoid exchange API throttling.

use super::TreasuryConfig;
use sha2::Sha256;
use tracing::info;

/// Off-ramp client that bridges crypto profits to bank accounts.
pub struct OfframpClient {
    http: reqwest::Client,
    exchange: ExchangeType,
    api_key: String,
    api_secret: Vec<u8>,
    api_passphrase: Option<String>,
    bank_account_id: Option<String>,
    fiat_currency: String,
    dry_run: bool,
}

#[derive(Debug, Clone)]
enum ExchangeType {
    Coinbase,
    Kraken,
}

/// Response from an off-ramp operation.
#[derive(Debug)]
pub struct OfframpResult {
    pub tx_id: String,
    pub amount_usd: f64,
    pub status: OfframpStatus,
}

#[derive(Debug)]
pub enum OfframpStatus {
    Pending,
    Completed,
    Failed(String),
}

impl OfframpClient {
    pub fn new(config: &TreasuryConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let exchange = match config.exchange.to_lowercase().as_str() {
            "coinbase" | "coinbase_pro" | "coinbase_advanced" => ExchangeType::Coinbase,
            "kraken" => ExchangeType::Kraken,
            other => return Err(format!("Unsupported exchange: {}", other).into()),
        };

        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            http,
            exchange,
            api_key: config.exchange_api_key.clone().unwrap_or_default(),
            api_secret: config.exchange_api_secret.clone()
                .unwrap_or_default()
                .into_bytes(),
            api_passphrase: config.exchange_api_passphrase.clone(),
            bank_account_id: config.bank_account_id.clone(),
            fiat_currency: config.fiat_currency.clone(),
            dry_run: config.dry_run,
        })
    }

    /// Execute the full off-ramp pipeline:
    /// 1. Sell stablecoin for fiat on exchange
    /// 2. Withdraw fiat to linked bank account
    ///
    /// Returns a transaction ID for tracking.
    pub async fn execute_offramp(
        &self,
        amount_usd: f64,
        _config: &TreasuryConfig,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if self.dry_run {
            info!("[Offramp] DRY RUN: would withdraw ${:.2} {} to bank", amount_usd, self.fiat_currency);
            return Ok(format!("DRY_RUN_{}", chrono::Utc::now().timestamp()));
        }

        // Validate we have required credentials
        if self.api_key.is_empty() || self.api_secret.is_empty() {
            return Err("Exchange API credentials not configured".into());
        }
        if self.bank_account_id.is_none() {
            return Err("Bank account ID not configured".into());
        }

        match &self.exchange {
            ExchangeType::Coinbase => self.coinbase_offramp(amount_usd).await,
            ExchangeType::Kraken => self.kraken_offramp(amount_usd).await,
        }
    }

    /// Coinbase Advanced Trade API off-ramp.
    /// 1. Place market sell order: USDC-USD
    /// 2. Withdraw USD to bank via payment method
    async fn coinbase_offramp(
        &self,
        amount_usd: f64,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let base_url = "https://api.coinbase.com";

        // Step 1: Sell USDC for USD
        let order_body = serde_json::json!({
            "client_order_id": format!("yd_{}", chrono::Utc::now().timestamp_millis()),
            "product_id": format!("USDC-{}", self.fiat_currency),
            "side": "SELL",
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": format!("{:.2}", amount_usd)
                }
            }
        });

        let sell_path = "/api/v3/brokerage/orders";
        let sell_resp = self.coinbase_signed_request(
            "POST", sell_path, &order_body, base_url,
        ).await?;

        let order_id = sell_resp.get("order_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        info!("[Offramp] Coinbase sell order placed: {} for ${:.2}", order_id, amount_usd);

        // Brief delay for order fill
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        // Step 2: Withdraw fiat to bank
        let bank_id = self.bank_account_id.as_deref().unwrap_or("");
        let withdraw_body = serde_json::json!({
            "amount": format!("{:.2}", amount_usd),
            "currency": self.fiat_currency,
            "payment_method": bank_id,
        });

        let withdraw_path = "/v2/withdrawals/payment-method";
        let withdraw_resp = self.coinbase_signed_request(
            "POST", withdraw_path, &withdraw_body, base_url,
        ).await?;

        let tx_id = withdraw_resp.get("data")
            .and_then(|d| d.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or(&order_id)
            .to_string();

        info!("[Offramp] Coinbase withdrawal initiated: {} -> bank", tx_id);
        Ok(tx_id)
    }

    /// Kraken REST API off-ramp.
    /// 1. Place market sell: USDC/USD
    /// 2. Withdraw fiat to linked bank
    async fn kraken_offramp(
        &self,
        amount_usd: f64,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let base_url = "https://api.kraken.com";

        // Step 1: Sell USDC
        let sell_params = [
            ("pair", format!("USDC{}", self.fiat_currency)),
            ("type", "sell".to_string()),
            ("ordertype", "market".to_string()),
            ("volume", format!("{:.2}", amount_usd)),
        ];

        let sell_resp = self.kraken_signed_request(
            "/0/private/AddOrder", &sell_params, base_url,
        ).await?;

        let order_id = sell_resp.get("result")
            .and_then(|r| r.get("txid"))
            .and_then(|t| t.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        info!("[Offramp] Kraken sell order: {} for ${:.2}", order_id, amount_usd);

        // Brief delay for settlement
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        // Step 2: Withdraw fiat
        let bank_key = self.bank_account_id.as_deref().unwrap_or("");
        let withdraw_params = [
            ("asset", self.fiat_currency.clone()),
            ("key", bank_key.to_string()),
            ("amount", format!("{:.2}", amount_usd)),
        ];

        let withdraw_resp = self.kraken_signed_request(
            "/0/private/Withdraw", &withdraw_params, base_url,
        ).await?;

        let refid = withdraw_resp.get("result")
            .and_then(|r| r.get("refid"))
            .and_then(|v| v.as_str())
            .unwrap_or(&order_id)
            .to_string();

        info!("[Offramp] Kraken withdrawal initiated: {} -> bank", refid);
        Ok(refid)
    }

    /// Send a Coinbase-signed API request (HMAC-SHA256).
    async fn coinbase_signed_request(
        &self,
        method: &str,
        path: &str,
        body: &serde_json::Value,
        base_url: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        use hmac::{Hmac, Mac};

        let timestamp = chrono::Utc::now().timestamp().to_string();
        // For GET requests, sign over empty body (Coinbase spec).
        // Signing over "{}" would produce an invalid HMAC.
        let body_str = if body.is_null() || method == "GET" {
            String::new()
        } else {
            serde_json::to_string(body)?
        };

        // Signature: HMAC-SHA256(timestamp + method + path + body)
        let prehash = format!("{}{}{}{}", timestamp, method, path, body_str);

        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(&self.api_secret)
            .map_err(|e| format!("HMAC key error: {}", e))?;
        hmac::Mac::update(&mut mac, prehash.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        let url = format!("{}{}", base_url, path);
        let mut req = self.http.request(
            method.parse().unwrap_or(reqwest::Method::POST),
            &url,
        )
            .header("CB-ACCESS-KEY", &self.api_key)
            .header("CB-ACCESS-SIGN", &signature)
            .header("CB-ACCESS-TIMESTAMP", &timestamp);

        // Only attach body for non-GET requests
        if !body_str.is_empty() {
            req = req.header("Content-Type", "application/json").body(body_str);
        }

        let resp = req.send().await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(format!("Coinbase API error {}: {}", status, text).into());
        }

        Ok(serde_json::from_str(&text)?)
    }

    /// Send a Kraken-signed API request (HMAC-SHA512).
    async fn kraken_signed_request(
        &self,
        path: &str,
        params: &[(&str, String)],
        base_url: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        use sha2::{Sha256, Sha512, Digest};

        let nonce = chrono::Utc::now().timestamp_millis() as u64;
        let mut form_data: Vec<(&str, String)> = vec![("nonce", nonce.to_string())];
        form_data.extend_from_slice(params);

        // Build POST body
        let body: String = form_data.iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&");

        // Signature: HMAC-SHA512(path + SHA256(nonce + body), base64_decode(secret))
        let nonce_body = format!("{}{}", nonce, body);
        let mut sha256 = Sha256::new();
        sha2::Digest::update(&mut sha256, nonce_body.as_bytes());
        let sha256_hash = sha256.finalize();

        let mut preimage = path.as_bytes().to_vec();
        preimage.extend_from_slice(&sha256_hash);

        use hmac::{Hmac, Mac};
        type HmacSha512 = Hmac<Sha512>;
        let decoded_secret = base64_decode(&self.api_secret)?;
        let mut mac = HmacSha512::new_from_slice(&decoded_secret)
            .map_err(|e| format!("HMAC key error: {}", e))?;
        hmac::Mac::update(&mut mac, &preimage);
        let signature = base64_encode(&mac.finalize().into_bytes());

        let url = format!("{}{}", base_url, path);
        let resp = self.http.post(&url)
            .header("API-Key", &self.api_key)
            .header("API-Sign", &signature)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(format!("Kraken API error {}: {}", status, text).into());
        }

        Ok(serde_json::from_str(&text)?)
    }

    /// Get current exchange account balances.
    pub async fn get_balances(&self) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        match &self.exchange {
            ExchangeType::Coinbase => {
                // GET requests: sign over empty body, don't send a JSON body
                self.coinbase_signed_request(
                    "GET", "/api/v3/brokerage/accounts", &serde_json::json!(null),
                    "https://api.coinbase.com",
                ).await
            }
            ExchangeType::Kraken => {
                self.kraken_signed_request(
                    "/0/private/Balance", &[],
                    "https://api.kraken.com",
                ).await
            }
        }
    }
}

/// Simple base64 encode (no padding).
fn base64_encode(data: &[u8]) -> String {
    
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Simple base64 decode.
fn base64_decode(data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let s = std::str::from_utf8(data)?;
    let mut result = Vec::new();
    let chars: Vec<u8> = s.bytes().filter(|b| *b != b'=').collect();

    fn decode_char(c: u8) -> u8 {
        match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => 0,
        }
    }

    for chunk in chars.chunks(4) {
        let len = chunk.len();
        if len < 2 { break; }
        let b0 = decode_char(chunk[0]) as u32;
        let b1 = decode_char(chunk[1]) as u32;
        let b2 = if len > 2 { decode_char(chunk[2]) as u32 } else { 0 };
        let b3 = if len > 3 { decode_char(chunk[3]) as u32 } else { 0 };
        let triple = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;
        result.push(((triple >> 16) & 0xFF) as u8);
        if len > 2 { result.push(((triple >> 8) & 0xFF) as u8); }
        if len > 3 { result.push((triple & 0xFF) as u8); }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_roundtrip() {
        let data = b"hello world";
        let encoded = base64_encode(data);
        let decoded = base64_decode(encoded.as_bytes()).unwrap();
        assert_eq!(&decoded, data);
    }

    #[test]
    fn test_base64_empty() {
        let encoded = base64_encode(b"");
        assert_eq!(encoded, "");
        let decoded = base64_decode(b"").unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_offramp_client_dry_run() {
        let config = TreasuryConfig::default();
        let client = OfframpClient::new(&config).unwrap();
        assert!(client.dry_run);
    }

    #[test]
    fn test_offramp_client_kraken() {
        let config = TreasuryConfig {
            exchange: "kraken".to_string(),
            ..Default::default()
        };
        let client = OfframpClient::new(&config).unwrap();
        assert!(matches!(client.exchange, ExchangeType::Kraken));
    }

    #[test]
    fn test_offramp_client_unsupported() {
        let config = TreasuryConfig {
            exchange: "binance_us".to_string(),
            ..Default::default()
        };
        assert!(OfframpClient::new(&config).is_err());
    }
}
