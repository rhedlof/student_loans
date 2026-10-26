import argparse
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def generate_avalanche_plot(
    file_path="Student_Loan_Payoff_Dashboard.xlsx",
    extra_monthly_budget=1000.00,
    strategy="avalanche",
    start_date=None,
):
    if start_date is None:
        start_date = datetime.now()

    # ----------------------------------------------------
    # 1. LOAD & PREPARE DATA
    # ----------------------------------------------------
    df = pd.read_excel(file_path, sheet_name="Loan Inventory")
    df = df.dropna(
        subset=["Loan Name", "Current Balance", "Est. Interest Rate"]
    )
    df = df[df["Loan Name"].astype(str).str.upper() != "TOTAL DEBT"].copy()

    df["Current Balance"] = pd.to_numeric(
        df["Current Balance"], errors="coerce"
    )

    if df["Est. Interest Rate"].dtype == object:
        df["Est. Interest Rate"] = (
            df["Est. Interest Rate"]
            .astype(str)
            .str.rstrip("%")
            .astype(float)
            / 100.0
        )
    elif df["Est. Interest Rate"].max() > 1.0:
        df["Est. Interest Rate"] = df["Est. Interest Rate"] / 100.0

    # Calculate a fallback minimum payment estimate (10-yr amortization)
    # for every loan, used to fill in the column when it's absent AND
    # to patch any individual blank/NaN cells in an existing column.
    # Leaving a NaN in "Minimum Payment" is dangerous: Python's min()
    # silently returns the non-NaN operand depending on argument order,
    # which can zero out balances unexpectedly during the simulation.
    r = df["Est. Interest Rate"] / 12.0
    n = 120
    estimated_minimum_payment = (
        df["Current Balance"] * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    )

    if "Minimum Payment" not in df.columns:
        df["Minimum Payment"] = estimated_minimum_payment
    else:
        df["Minimum Payment"] = pd.to_numeric(
            df["Minimum Payment"], errors="coerce"
        )
        missing_mask = df["Minimum Payment"].isna()
        if missing_mask.any():
            df.loc[missing_mask, "Minimum Payment"] = estimated_minimum_payment[
                missing_mask
            ]

    all_loans_raw = df.to_dict("records")

    # Determine Priority Order based on strategy
    if strategy.lower() == "hybrid":
        priv_sorted = sorted(
            [l for l in all_loans_raw if l.get("Type", "").lower() == "private"],
            key=lambda x: x["Est. Interest Rate"],
            reverse=True,
        )
        fed_sorted = sorted(
            [l for l in all_loans_raw if l.get("Type", "").lower() == "federal"],
            key=lambda x: x["Est. Interest Rate"],
            reverse=True,
        )
        priority_ordered_loans = priv_sorted + fed_sorted
        strategy_desc = "Hybrid (Private -> Federal)"
    else:
        priority_ordered_loans = sorted(
            all_loans_raw, key=lambda x: x["Est. Interest Rate"], reverse=True
        )
        strategy_desc = "Pure Avalanche (Highest Interest First)"

    priority_map = {
        l["Loan Name"]: rank + 1
        for rank, l in enumerate(priority_ordered_loans)
    }

    # ----------------------------------------------------
    # 2. AMORTIZATION SIMULATION
    # ----------------------------------------------------
    loans = [dict(l) for l in all_loans_raw]
    month = 0
    history = []
    individual_payoff_months = {}
    total_interest_paid = 0.0
    focus_payment_history = []

    while month < 360:
        total_bal = sum(l["Current Balance"] for l in loans)
        if total_bal <= 0.01:
            break

        month += 1
        snapshot = {"Month": month}
        for l in loans:
            snapshot[l["Loan Name"]] = max(0.0, l["Current Balance"])
        history.append(snapshot)

        # Accrue monthly interest
        for l in loans:
            if l["Current Balance"] > 0:
                interest_accrued = l["Current Balance"] * (
                    l["Est. Interest Rate"] / 12.0
                )
                l["Current Balance"] += interest_accrued
                total_interest_paid += interest_accrued

        # Minimum payments — any minimum payment freed up by an
        # already-paid-off loan gets rolled into the extra pool below,
        # which is what makes this a "true" debt avalanche (minimums
        # from retired loans snowball onto the next target debt).
        freed_minimum_payments = 0.0
        minimum_paid_this_month = {}
        for l in loans:
            if l["Current Balance"] > 0:
                pmt = min(l["Current Balance"], l["Minimum Payment"])
                l["Current Balance"] -= pmt
                minimum_paid_this_month[l["Loan Name"]] = pmt
            else:
                freed_minimum_payments += l["Minimum Payment"]

        # Target extra payment
        extra_pool = extra_monthly_budget + freed_minimum_payments
        if strategy.lower() == "hybrid":
            active_priv = sorted(
                [
                    l
                    for l in loans
                    if l.get("Type", "").lower() == "private"
                    and l["Current Balance"] > 0
                ],
                key=lambda x: x["Est. Interest Rate"],
                reverse=True,
            )
            active_fed = sorted(
                [
                    l
                    for l in loans
                    if l.get("Type", "").lower() == "federal"
                    and l["Current Balance"] > 0
                ],
                key=lambda x: x["Est. Interest Rate"],
                reverse=True,
            )
            target_list = active_priv + active_fed
        else:
            target_list = sorted(
                [l for l in loans if l["Current Balance"] > 0],
                key=lambda x: x["Est. Interest Rate"],
                reverse=True,
            )

        # The "focus" loan is whichever active loan sits at the top of
        # this month's priority order — the one all extra dollars get
        # aimed at first.
        focus_name = target_list[0]["Loan Name"] if target_list else None

        extra_paid_this_month = {}
        for l in target_list:
            if extra_pool <= 0:
                break
            if l["Current Balance"] > 0:
                extra_pmt = min(l["Current Balance"], extra_pool)
                l["Current Balance"] -= extra_pmt
                extra_pool -= extra_pmt
                extra_paid_this_month[l["Loan Name"]] = extra_pmt

        if focus_name is not None:
            focus_total_payment = minimum_paid_this_month.get(
                focus_name, 0.0
            ) + extra_paid_this_month.get(focus_name, 0.0)
            focus_payment_history.append(
                {
                    "Month": month,
                    "Loan Name": focus_name,
                    "Payment": focus_total_payment,
                }
            )

        for l in loans:
            name = l["Loan Name"]
            if (
                l["Current Balance"] <= 0.01
                and name not in individual_payoff_months
            ):
                individual_payoff_months[name] = month

    snapshot = {"Month": month}
    for l in loans:
        snapshot[l["Loan Name"]] = 0.0
    history.append(snapshot)

    history_df = pd.DataFrame(history)

    total_payoff_date = start_date + pd.DateOffset(months=month)
    total_payoff_str = total_payoff_date.strftime("%B %Y")

    individual_payoff_dates = {}
    for name, p_month in individual_payoff_months.items():
        p_date = start_date + pd.DateOffset(months=p_month)
        individual_payoff_dates[name] = p_date.strftime("%b %Y")

    # ----------------------------------------------------
    # 3. PLOT GRAPH
    # ----------------------------------------------------
    # Stack loans in payoff-priority order (rank #1 sits right on the
    # x-axis, rank #N forms the outer edge) instead of raw spreadsheet
    # row order. This makes each band collapse to the axis in the same
    # sequence money is actually applied, so the chart reads left-to-
    # right AND bottom-to-top as a payoff cascade, and the stacking
    # order now matches the legend order below.
    loan_names = [
        name
        for name, _rank in sorted(priority_map.items(), key=lambda kv: kv[1])
    ]
    num_loans = len(loan_names)

    # Color by priority rank (not raw interest rate) so the gradient is
    # always a clean "paid first -> paid last" ramp, including under the
    # hybrid strategy where rank and rate can diverge.
    norm = mcolors.Normalize(vmin=1, vmax=num_loans)
    colormap = cm.YlOrRd_r
    get_loan_color = lambda name: colormap(norm(priority_map[name]))

    fig, ax = plt.subplots(figsize=(14, 8))

    y_stack = np.zeros((len(history_df), num_loans))
    for i, name in enumerate(loan_names):
        y_stack[:, i] = history_df[name].values

    y_cumulative = np.cumsum(y_stack, axis=1)
    x = history_df["Month"].values

    legend_entries = []

    for i, name in enumerate(loan_names):
        y_lower = y_cumulative[:, i - 1] if i > 0 else np.zeros_like(x)
        y_upper = y_cumulative[:, i]

        matching_loan = next(l for l in all_loans_raw if l["Loan Name"] == name)
        rate_pct = matching_loan["Est. Interest Rate"] * 100.0
        rank = priority_map[name]
        payoff_date_text = individual_payoff_dates.get(name, "N/A")

        curve_color = get_loan_color(name)
        label_text = (
            f"#{rank}: {name} ({rate_pct:.2f}%) — Paid: {payoff_date_text}"
        )

        is_top_curve = i == num_loans - 1
        line_width = 3.2 if is_top_curve else 0.75
        line_alpha = 1.0 if is_top_curve else 0.8

        ax.plot(
            x, y_upper, color="black", linewidth=line_width, alpha=line_alpha
        )
        poly = ax.fill_between(
            x,
            y_lower,
            y_upper,
            color=curve_color,
            alpha=0.5,
            label=label_text,
        )

        legend_entries.append(
            {"handle": poly, "label": label_text, "priority_rank": rank}
        )

    title_line1 = f"Loan Payoff Trajectory ({num_loans} Individual Loans)"
    title_line2 = f"DEBT FREE DATE: {total_payoff_str.upper()} ({month} Months) | Strategy: {strategy_desc}"
    title_line3 = f"Total Interest Paid: USD {total_interest_paid:,.2f} | Extra Budget: +USD {extra_monthly_budget:,.0f}/mo"

    ax.set_title(
        f"{title_line1}\n{title_line2}\n{title_line3}",
        fontsize=11,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Time (Months)", fontsize=11)
    ax.set_ylabel("Total Remaining Principal ($)", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(True, linestyle="--", alpha=0.4)

    sorted_entries = sorted(legend_entries, key=lambda e: e["priority_rank"])
    sorted_handles = [e["handle"] for e in sorted_entries]
    sorted_labels = [e["label"] for e in sorted_entries]

    ax.legend(
        sorted_handles,
        sorted_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=6.5 if num_loans > 20 else 7.5,
        ncol=1,
        frameon=True,
        title="Payoff Priority Order & Completion Date",
        title_fontsize=8,
    )

    plt.tight_layout()
    output_filename = "avalanche_payoff_chart.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # ----------------------------------------------------
    # 4. FOCUS-LOAN MONTHLY PAYMENT CHART
    # ----------------------------------------------------
    focus_df = pd.DataFrame(focus_payment_history)

    fig2, ax2 = plt.subplots(figsize=(14, 6))

    bar_colors = [
        get_loan_color(name) for name in focus_df["Loan Name"]
    ]
    ax2.bar(
        focus_df["Month"],
        focus_df["Payment"],
        color=bar_colors,
        width=1.0,
        edgecolor="none",
    )

    # Label each contiguous block of months spent focused on the same
    # loan, once, roughly centered over that block.
    focus_df["block"] = (
        focus_df["Loan Name"] != focus_df["Loan Name"].shift()
    ).cumsum()
    for _, block_df in focus_df.groupby("block"):
        loan_name = block_df["Loan Name"].iloc[0]
        rank = priority_map[loan_name]
        mid_month = block_df["Month"].mean()
        max_payment = block_df["Payment"].max()
        ax2.annotate(
            f"#{rank}: {loan_name}",
            xy=(mid_month, max_payment),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90 if len(block_df) < 3 else 0,
        )

    ax2.set_title(
        "Monthly Payment to Current Focus Loan\n"
        f"(Minimum + Extra Applied to Whichever Loan is Top Priority That Month | Strategy: {strategy_desc})",
        fontsize=11,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xlabel("Time (Months)", fontsize=11)
    ax2.set_ylabel("Payment to Focus Loan ($)", fontsize=11)
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    focus_output_filename = "focus_loan_payment_chart.png"
    plt.savefig(focus_output_filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate and plot loan payoff trajectories."
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="Student_Loan_Payoff_Dashboard.xlsx",
        help="Path to Excel sheet",
    )
    parser.add_argument(
        "-e",
        "--extra",
        type=float,
        default=1000.0,
        help="Extra monthly payment budget (default: 1000)",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="avalanche",
        choices=["avalanche", "hybrid"],
        help="Payoff strategy: 'avalanche' or 'hybrid' (default: avalanche)",
    )

    args = parser.parse_args()

    generate_avalanche_plot(
        file_path=args.file,
        extra_monthly_budget=args.extra,
        strategy=args.strategy,
    )
