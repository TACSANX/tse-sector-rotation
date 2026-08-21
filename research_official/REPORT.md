# TOPIX-17 Robustness Research — Official NEXT FUNDS Data

## Data

Official Nomura Asset Management historical CSVs; distribution-reinvested NAV per share.
Common research horizon: 2008-03-21 to 2026-08-20.

## Equal-weight benchmark

- Full CAGR: 11.98%
- Full max drawdown: -35.45%
- Full Sharpe: 0.711
- Holdout CAGR: 11.95%
- Holdout max drawdown: -35.45%
- Holdout Sharpe: 0.737

## Best variants by 2018+ holdout Sharpe

| variant                      |   full_cagr |   full_max_drawdown |   full_sharpe |   pre2018_cagr |   pre2018_sharpe |   holdout_cagr |   holdout_max_drawdown |   holdout_sharpe |   cash_exposure |   annualized_turnover |
|:-----------------------------|------------:|--------------------:|--------------:|---------------:|-----------------:|---------------:|-----------------------:|-----------------:|----------------:|----------------------:|
| monthly_top3_current_0bp     |      0.0959 |             -0.4072 |        0.5512 |         0.0630 |           0.4053 |         0.1300 |                -0.3076 |           0.6941 |          0.0874 |               13.4392 |
| monthly_top3_none_0bp        |      0.1054 |             -0.3820 |        0.5804 |         0.0843 |           0.4928 |         0.1271 |                -0.3820 |           0.6662 |          0.0000 |               12.4496 |
| monthly_top3_current_5bp     |      0.0887 |             -0.4211 |        0.5194 |         0.0555 |           0.3705 |         0.1233 |                -0.3124 |           0.6652 |          0.0874 |               13.4392 |
| monthly_top3_risk_off_0bp    |      0.0967 |             -0.3629 |        0.5698 |         0.0807 |           0.4985 |         0.1133 |                -0.2819 |           0.6398 |          0.1718 |               14.3101 |
| monthly_top3_none_5bp        |      0.0987 |             -0.3873 |        0.5518 |         0.0775 |           0.4628 |         0.1206 |                -0.3873 |           0.6391 |          0.0000 |               12.4496 |
| monthly_top3_current_10bp    |      0.0816 |             -0.4347 |        0.4875 |         0.0481 |           0.3357 |         0.1165 |                -0.3172 |           0.6362 |          0.0874 |               13.4392 |
| monthly_top3_none_10bp       |      0.0920 |             -0.3926 |        0.5233 |         0.0707 |           0.4328 |         0.1140 |                -0.3926 |           0.6120 |          0.0000 |               12.4496 |
| monthly_top3_risk_off_5bp    |      0.0891 |             -0.3778 |        0.5345 |         0.0728 |           0.4607 |         0.1059 |                -0.2857 |           0.6068 |          0.1718 |               14.3101 |
| monthly_top3_current_20bp    |      0.0675 |             -0.4609 |        0.4237 |         0.0334 |           0.2659 |         0.1031 |                -0.3268 |           0.5781 |          0.0874 |               13.4392 |
| monthly_top3_risk_off_10bp   |      0.0816 |             -0.3924 |        0.4991 |         0.0649 |           0.4230 |         0.0987 |                -0.2895 |           0.5737 |          0.1718 |               14.3101 |
| monthly_top3_dual_trend_0bp  |      0.0650 |             -0.4533 |        0.4375 |         0.0391 |           0.3050 |         0.0918 |                -0.2661 |           0.5665 |          0.2936 |               12.2121 |
| monthly_top3_none_20bp       |      0.0788 |             -0.4031 |        0.4661 |         0.0573 |           0.3727 |         0.1011 |                -0.4031 |           0.5576 |          0.0000 |               12.4496 |
| monthly_top1_current_0bp     |      0.0836 |             -0.4264 |        0.4520 |         0.0529 |           0.3396 |         0.1156 |                -0.3260 |           0.5554 |          0.0874 |               17.5165 |
| weekly_top3_none_0bp         |      0.0966 |             -0.3576 |        0.5510 |         0.0999 |           0.5635 |         0.0934 |                -0.3576 |           0.5383 |          0.0000 |               27.2970 |
| monthly_top3_dual_trend_5bp  |      0.0587 |             -0.4626 |        0.4051 |         0.0330 |           0.2723 |         0.0853 |                -0.2664 |           0.5345 |          0.2936 |               12.2121 |
| monthly_top1_current_5bp     |      0.0745 |             -0.4366 |        0.4176 |         0.0427 |           0.2981 |         0.1075 |                -0.3314 |           0.5274 |          0.0874 |               17.5165 |
| monthly_top3_current_30bp    |      0.0536 |             -0.4860 |        0.3598 |         0.0188 |           0.1962 |         0.0898 |                -0.3362 |           0.5200 |          0.0874 |               13.4392 |
| monthly_top1_none_0bp        |      0.0782 |             -0.4611 |        0.4270 |         0.0532 |           0.3381 |         0.1041 |                -0.3852 |           0.5090 |          0.0000 |               17.1602 |
| monthly_top3_risk_off_20bp   |      0.0666 |             -0.4206 |        0.4282 |         0.0494 |           0.3474 |         0.0842 |                -0.2971 |           0.5074 |          0.1718 |               14.3101 |
| monthly_top3_none_30bp       |      0.0658 |             -0.4134 |        0.4088 |         0.0440 |           0.3126 |         0.0883 |                -0.4134 |           0.5032 |          0.0000 |               12.4496 |
| monthly_top3_dual_trend_10bp |      0.0524 |             -0.4717 |        0.3727 |         0.0269 |           0.2396 |         0.0788 |                -0.2666 |           0.5025 |          0.2936 |               12.2121 |
| monthly_top1_current_10bp    |      0.0653 |             -0.4467 |        0.3832 |         0.0326 |           0.2566 |         0.0994 |                -0.3367 |           0.4993 |          0.0874 |               17.5165 |
| monthly_top1_none_5bp        |      0.0692 |             -0.4712 |        0.3942 |         0.0433 |           0.2993 |         0.0961 |                -0.3919 |           0.4817 |          0.0000 |               17.1602 |
| weekly_top3_none_5bp         |      0.0821 |             -0.3702 |        0.4871 |         0.0860 |           0.5024 |         0.0784 |                -0.3702 |           0.4715 |          0.0000 |               27.2970 |
| monthly_top1_none_10bp       |      0.0604 |             -0.4812 |        0.3614 |         0.0336 |           0.2604 |         0.0881 |                -0.3986 |           0.4543 |          0.0000 |               17.1602 |

## Interpretation rule

Do not select the first row mechanically. Prefer parameter neighborhoods where monthly/weekly, Top1/Top3 and cost assumptions tell a consistent story, and where both pre-2018 and 2018+ results remain acceptable. A single isolated optimum is treated as overfit.
