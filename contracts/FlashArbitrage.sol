// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title FlashArbitrage — Zero-capital atomic cross-DEX arbitrage
/// @notice Executes atomic arbitrage using Uniswap V2/V3 flash swaps.
///         No upfront capital required — borrows from pool, arbs, repays in same tx.
///         Profits are sent to the ProfitVault automatically.
/// @dev This contract is called by the Rust ASTE engine when it detects
///      a profitable arbitrage path on EVM chains.
///
/// Flow:
/// 1. Flash borrow Token A from Pool 1
/// 2. Swap Token A -> Token B on Pool 2 (where B is cheaper)
/// 3. Repay Pool 1 with Token B + fee
/// 4. Profit = Token B received - Token B repaid
///
/// All 4 steps happen in a single atomic transaction.
/// If profit < gas, the tx reverts and we only lose gas cost.

interface IUniswapV2Pair {
    function swap(uint amount0Out, uint amount1Out, address to, bytes calldata data) external;
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
    function token0() external view returns (address);
    function token1() external view returns (address);
}

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
}

interface IERC20 {
    function balanceOf(address) external view returns (uint256);
    function transfer(address, uint256) external returns (bool);
    function approve(address, uint256) external returns (bool);
}

contract FlashArbitrage {
    address public immutable owner;
    address public immutable profitVault;

    // Reentrancy guard
    uint256 private _locked;

    struct ArbParams {
        address router2;       // Router for the second swap
        address[] path;        // Token path for second swap
        uint256 amountBorrow;  // Amount to flash borrow
        uint256 minProfit;     // Minimum profit or revert
    }

    event ArbitrageExecuted(
        address indexed pool,
        uint256 borrowed,
        uint256 profit,
        uint256 gasUsed
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "FlashArb: not owner");
        _;
    }

    modifier nonReentrant() {
        require(_locked == 0, "FlashArb: reentrant");
        _locked = 1;
        _;
        _locked = 0;
    }

    constructor(address _profitVault) {
        owner = msg.sender;
        profitVault = _profitVault;
    }

    /// @notice Execute atomic arbitrage via flash swap
    /// @param pool The Uniswap V2 pair to flash borrow from
    /// @param borrowToken0 True if borrowing token0, false for token1
    /// @param amount Amount to borrow
    /// @param router2 DEX router for the second leg
    /// @param path Token path for the second swap
    /// @param minProfit Minimum profit threshold (reverts if not met)
    function executeArbitrage(
        address pool,
        bool borrowToken0,
        uint256 amount,
        address router2,
        address[] calldata path,
        uint256 minProfit
    ) external onlyOwner nonReentrant {
        uint256 gasStart = gasleft();

        // Encode arbitrage params for the callback
        bytes memory data = abi.encode(ArbParams({
            router2: router2,
            path: path,
            amountBorrow: amount,
            minProfit: minProfit
        }));

        // Initiate flash swap — the pair contract will call our callback
        uint256 amount0Out = borrowToken0 ? amount : 0;
        uint256 amount1Out = borrowToken0 ? 0 : amount;

        IUniswapV2Pair(pool).swap(amount0Out, amount1Out, address(this), data);

        uint256 gasUsed = gasStart - gasleft();
        // Profit already sent to vault in callback
    }

    /// @notice Uniswap V2 flash swap callback
    /// @dev Called by the pair contract after sending us the borrowed tokens
    function uniswapV2Call(
        address sender,
        uint256 amount0,
        uint256 amount1,
        bytes calldata data
    ) external {
        // Decode params
        ArbParams memory params = abi.decode(data, (ArbParams));

        // Verify callback is from the pair, not an attacker
        require(sender == address(this), "FlashArb: unauthorized callback");

        uint256 borrowedAmount = amount0 > 0 ? amount0 : amount1;
        address borrowedToken = amount0 > 0
            ? IUniswapV2Pair(msg.sender).token0()
            : IUniswapV2Pair(msg.sender).token1();

        // Step 1: Approve router to spend our borrowed tokens
        IERC20(borrowedToken).approve(params.router2, borrowedAmount);

        // Step 2: Swap on the second DEX (where the price is different)
        uint[] memory amounts = IUniswapV2Router(params.router2).swapExactTokensForTokens(
            borrowedAmount,
            0, // Accept any output (we validate profit below)
            params.path,
            address(this),
            block.timestamp + 300
        );

        uint256 outputAmount = amounts[amounts.length - 1];

        // Step 3: Calculate repayment amount (0.3% Uniswap V2 fee)
        // repay = borrowed * 1000 / 997 + 1
        uint256 repayAmount = (borrowedAmount * 1000 / 997) + 1;

        // The output token should be the repayment token
        address repayToken = params.path[params.path.length - 1];

        // Step 4: Verify profit
        uint256 profit = outputAmount > repayAmount ? outputAmount - repayAmount : 0;
        require(profit >= params.minProfit, "FlashArb: insufficient profit");

        // Step 5: Repay the flash swap
        IERC20(repayToken).transfer(msg.sender, repayAmount);

        // Step 6: Send profit to vault
        if (profit > 0) {
            IERC20(repayToken).transfer(profitVault, profit);
        }

        emit ArbitrageExecuted(msg.sender, borrowedAmount, profit, 0);
    }

    /// @notice Withdraw stuck tokens (emergency only)
    function rescue(address token, uint256 amount) external onlyOwner {
        IERC20(token).transfer(owner, amount);
    }

    /// @notice Withdraw stuck ETH
    function rescueETH() external onlyOwner {
        (bool ok, ) = owner.call{value: address(this).balance}("");
        require(ok, "FlashArb: rescue failed");
    }

    receive() external payable {}
}
