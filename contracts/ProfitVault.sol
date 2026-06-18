// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ProfitVault — Autonomous yield accumulation vault
/// @notice Collects arbitrage profits from the ASTE engine and provides
///         owner-only batch withdrawal to exchange deposit addresses.
/// @dev Designed for Arbitrum/Base/Polygon to minimize gas costs.
///      The vault is the on-chain "safe" where profits accumulate between
///      off-ramp cycles. Only the deployer (engine operator) can withdraw.
contract ProfitVault {
    address public immutable owner;

    uint256 public totalDeposited;
    uint256 public totalWithdrawn;
    uint256 public depositCount;
    uint256 public withdrawCount;

    // Reentrancy guard
    uint256 private _locked;

    event ProfitDeposited(uint256 amount, uint256 totalDeposited);
    event FundsWithdrawn(address indexed to, uint256 amount);
    event TokenWithdrawn(address indexed token, address indexed to, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "ProfitVault: not owner");
        _;
    }

    modifier nonReentrant() {
        require(_locked == 0, "ProfitVault: reentrant");
        _locked = 1;
        _;
        _locked = 0;
    }

    constructor() {
        owner = msg.sender;
    }

    /// @notice Receive ETH profits from arbitrage bundles
    receive() external payable {
        totalDeposited += msg.value;
        depositCount++;
        emit ProfitDeposited(msg.value, totalDeposited);
    }

    /// @notice Explicit deposit function (also accepts ETH)
    function deposit() external payable {
        require(msg.value > 0, "ProfitVault: zero deposit");
        totalDeposited += msg.value;
        depositCount++;
        emit ProfitDeposited(msg.value, totalDeposited);
    }

    /// @notice Withdraw all ETH to owner (batch off-ramp trigger)
    function withdrawAll() external onlyOwner nonReentrant {
        uint256 balance = address(this).balance;
        require(balance > 0, "ProfitVault: empty");
        totalWithdrawn += balance;
        withdrawCount++;
        (bool ok, ) = owner.call{value: balance}("");
        require(ok, "ProfitVault: transfer failed");
        emit FundsWithdrawn(owner, balance);
    }

    /// @notice Withdraw specific amount to a target address
    ///         (used for sending directly to exchange deposit address)
    function withdrawTo(address payable to, uint256 amount) external onlyOwner nonReentrant {
        require(amount > 0 && amount <= address(this).balance, "ProfitVault: bad amount");
        totalWithdrawn += amount;
        withdrawCount++;
        (bool ok, ) = to.call{value: amount}("");
        require(ok, "ProfitVault: transfer failed");
        emit FundsWithdrawn(to, amount);
    }

    /// @notice Withdraw ERC20 tokens (for stablecoin extraction)
    function withdrawToken(
        address token,
        address to,
        uint256 amount
    ) external onlyOwner nonReentrant {
        require(amount > 0, "ProfitVault: zero amount");
        (bool ok, bytes memory data) = token.call(
            abi.encodeWithSelector(0xa9059cbb, to, amount) // transfer(address,uint256)
        );
        require(ok && (data.length == 0 || abi.decode(data, (bool))), "ProfitVault: token transfer failed");
        emit TokenWithdrawn(token, to, amount);
    }

    /// @notice Get current ETH balance
    function getBalance() external view returns (uint256) {
        return address(this).balance;
    }

    /// @notice Get ERC20 token balance
    function getTokenBalance(address token) external view returns (uint256) {
        (bool ok, bytes memory data) = token.staticcall(
            abi.encodeWithSelector(0x70a08231, address(this)) // balanceOf(address)
        );
        if (!ok || data.length < 32) return 0;
        return abi.decode(data, (uint256));
    }

    /// @notice Get vault statistics
    function getStats() external view returns (
        uint256 balance,
        uint256 deposited,
        uint256 withdrawn,
        uint256 deposits,
        uint256 withdrawals
    ) {
        return (address(this).balance, totalDeposited, totalWithdrawn, depositCount, withdrawCount);
    }
}
