import pandas as pd
from scipy import stats


def chi2result(result):
    """
    The H0 (Null Hypothesis): There is no relationship between variable one and variable two.
    The H1 (Alternative Hypothesis): There is a relationship between variable 1 and variable 2.
    How to accept (or reject) H0 (Null hypothesis)?
    1. Determine if the results of test can be trusted; the assumption here is that at least
    20% of the expected values in the resulting matrix are > 5.0
    2. If the results of the test can be trusted (#1 above satisfied) then do the following:
    a) p-value is less than 0.05 then reject H0 (Null hypothesis)  --> v1 and v2 are related
    b) Otherwise accept H0.                                        --> v1 and v2 are independent.
    :param result:
    :return:
    """
    gt5count = sum = 0
    for e in result[3].flatten():
        sum += 1
        if e >= 5:
            gt5count = gt5count + 1
    ep = gt5count / sum
    if ep > .2:
        if result[1] < 0.05:
            return 'Categorical variables are related'
        return 'Categorical variables are independent'
    return 'A relationship between categorical variables cannot be determined'


if __name__ == "__main__":
    # Change the path as per your local directory
    df = pd.read_csv('/Volumes/G-DRIVE mobile/Data/FannieMae/2017Q1/Acquisition_2017Q1.txt',
                     sep='|', index_col=False,
                     names=['loan_identifier', 'channel', 'seller_name', 'original_interest_rate', 'original_upb',
                            'original_loan_term', 'origination_date', 'first_paymane_date', 'ltv', 'cltv',
                            'number_of_borrowers', 'dti', 'borrower_credit_score',
                            'first_time_home_buyer_indicator', 'loan_purpose', 'property_type', 'number_of_units',
                            'occupancy_status', 'property_state', 'zip_3_digit', 'mortgage_insurance_percentage',
                            'product_type', 'co_borrower_credit_score', 'mortgage_insurance_type',
                            'relocation_mortgage_indicator'])

    crosstab = pd.crosstab(df['mortgage_insurance_type'], df['property_type'])

    result = stats.chi2_contingency(crosstab)
    print('Chi-square statistic: ', result[0], ' p-value: ', result[1], ' Degrees of freedom: ', result[2])

    print(chi2result(result))
