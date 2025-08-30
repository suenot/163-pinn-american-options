# PINN for American Options -- Explained Simply

## What Is an American Option?

Imagine you bought a **coupon** that lets you return an item to a store **at any time** before it expires.

- You buy a coupon for $5 that says: "You can sell your old phone back to the store for $100 at any point in the next year."
- If your phone's market value drops to $60, you can use the coupon right now and get $100 (profit of $40 minus the $5 coupon cost).
- But maybe you should wait... what if the phone drops to $30 next month? Then you'd get $70 profit instead!
- However, if you wait too long and the phone price goes back up to $110, your coupon becomes worthless.

This is exactly how an **American put option** works:
- The "phone" is a stock or Bitcoin
- The "coupon" is the option contract
- The "$100 return price" is the **strike price**
- "Any time before it expires" is the **American-style** feature

**European options** are like coupons that only work on the exact expiration date. American options are more flexible (and more valuable) because you can use them whenever you want.

## The Big Question: When Should You Use the Coupon?

This is the **early exercise problem**. It is the hardest part of pricing American options.

Think of it like this:

> You have an ice cream coupon that expires in summer. Ice cream prices go up and down. When is the perfect time to use it?

- Use it too early: you might miss an even better deal later
- Wait too long: the coupon expires worthless, or prices go back up

There is a **magic line** called the **exercise boundary** that tells you: "If the price falls below this line, use your coupon NOW."

```
Price
  ^
  |     Don't use coupon yet (wait)
  |  . . . . . . . . . . . . . . . . .  <-- magic line
  |     USE YOUR COUPON NOW!
  +-----------------------------------> Time
  Today                           Expiry
```

## What Is a PINN?

**PINN** stands for **Physics-Informed Neural Network**.

Think of a regular neural network as a student who learns by looking at examples:
- "Here are 1000 photos of cats. Learn what a cat looks like."

A PINN is a student who also knows the **rules of the game**:
- "Learn what the option price looks like, BUT you must also follow these rules of finance (the Black-Scholes equation)."

It is like the difference between:
- Learning to play chess by watching 1000 games (regular neural network)
- Learning to play chess by watching 1000 games AND knowing the rules of how each piece moves (PINN)

The "physics" in PINN comes from the rules. In our case, the rule is the **Black-Scholes equation** -- a famous formula that describes how option prices change over time.

## How Does the PINN Learn?

The PINN has a **report card** with four grades:

### Grade 1: Following the Rules (PDE Loss)
"Did the network's answer follow the Black-Scholes equation?"

This is like checking if a student's math homework follows the correct formulas.

### Grade 2: Getting the Final Answer Right (Terminal Condition)
"At the expiration date, did the network give the correct price?"

At expiry, we KNOW the exact price: it is just max(strike - stock_price, 0) for a put. So we check if the network gets this right.

### Grade 3: Getting the Edges Right (Boundary Conditions)
"When the stock price is very high or very low, did the network give reasonable answers?"

- If the stock is at $0, a put option is worth the full strike price.
- If the stock is at $1,000,000, a put option with strike $100 is worthless.

### Grade 4: Respecting the Coupon Rule (Penalty Loss)
"The option price should NEVER be less than what you'd get by using the coupon right now."

If the coupon lets you sell at $100 and the stock is at $70, the option must be worth at least $30 (since you could exercise it right now for $30). If the network says $25, that is WRONG -- you would just exercise!

We give the network a **penalty** (like detention in school) whenever it breaks this rule.

## The Magic of Automatic Differentiation

One of the coolest things about PINNs is how they compute **Greeks** (the sensitivities of the option price).

Imagine you want to know: "If the stock price goes up by $1, how much does my option price change?"

Traditional methods: bump the stock price by $1, recalculate everything, find the difference. Slow and noisy.

PINN method: since the option price is computed by a neural network (which is just a chain of math operations), we can use the **chain rule of calculus** to get the exact answer instantly. This is called **automatic differentiation** or **autograd**.

It is like the difference between:
- Measuring the slope of a hill by walking up 1 meter and checking your altitude change (finite differences)
- Knowing the exact mathematical formula for the hill and computing the slope directly (autograd)

## Real-World Analogy: The Parking Meter

Here is another way to think about it:

You put money in a parking meter (= buy the option). The meter gives you the right to park (= hold the underlying) until it expires.

- **European meter**: You MUST park for exactly 2 hours. No refund if you leave early.
- **American meter**: You can leave anytime and get a partial refund based on unused time.

The question "how much is the American parking meter worth?" depends on:
- How likely you are to want to leave early (= stock price movement)
- How much refund you get (= exercise value)
- How much time is left on the meter (= time to expiry)

The PINN learns to answer this question for any combination of these factors, all at once.

## How We Use Real Market Data

We connect to **Bybit** (a cryptocurrency exchange) to get real Bitcoin and Ethereum prices. We also use stock data from services like Yahoo Finance.

Why? Because the PINN needs to know the **volatility** (how wildly the price moves) to price options correctly. Real data gives us real volatility.

- Bitcoin: very volatile (~60-80% per year) -- like a rollercoaster
- Apple stock: less volatile (~20-30% per year) -- like a gentle hill
- Higher volatility = more valuable options (more chance of big moves)

## Summary for a 10-Year-Old

1. An **option** is like a coupon to buy or sell something at a fixed price
2. An **American option** lets you use the coupon any time (not just at the end)
3. The hard part is figuring out **when** to use it and **how much** it is worth
4. We teach a **neural network** the rules of finance (PINN)
5. The network learns the right price AND the best time to use the coupon
6. It works for stocks, Bitcoin, and any other financial asset
7. Once trained, it gives answers **instantly** -- perfect for real-time trading
