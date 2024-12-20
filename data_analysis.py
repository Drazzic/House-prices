import pandas as pd
from scipy.stats import zscore

def data_analysis(df_cat, df_num, label):

    df_cat_analyse = pd.concat([df_cat, label], axis=1)

    for feature in df_cat_analyse.columns:
    # Get the distribution
        distribution = df_cat_analyse.groupby(feature)['SalePrice'].describe()
    
    # Save to CSV
    csv_filename = f"{feature}_price_distribution.csv"
    distribution.to_csv(csv_filename)
    print(f"Distribution for {feature} saved to {csv_filename}")


    df_num_analyse = pd.concat([df_num, label], axis=1)
    z_scores = df_num_analyse.apply(zscore) 
    # Display the resulting DataFrame
    print('The num feat z-scores is saved to cat_feat_ranking.csv')
    z_scores.to_csv('num_feat_zscores.csv', index=False)
    print()