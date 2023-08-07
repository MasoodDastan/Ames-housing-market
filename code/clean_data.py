   
import numpy as np
import pandas as pd

def create_infrequent_category(data, threshold, category_labels_dict={}):
    cat_dict = {}
    if category_labels_dict:
        for col in data.columns:
            data[col] = np.where(data[col].isin(category_labels_dict[col]), data[col], 'infrequent')
        
    else:
        for col in data.columns:  
            value_counts = data[col].value_counts()
            low_freq_levels = value_counts[value_counts < threshold].index
            data[col] = np.where(data[col].isin(low_freq_levels), 'infrequent', data[col])
            var_labels = data[col].value_counts().index.tolist()
            cat_dict[col]=var_labels
    return data, cat_dict

def clean_data(data, version, cat_labels={}):
    """
    Cleans the given dataset based on the specified version.

    Inputs:
    - data: DataFrame containing the dataset to be cleaned.
    - version: String indicating the version of cleaning to be applied ('0' or 'mean').

    Returns:
    - Cleaned DataFrame with missing values replaced according to the specified version.
    """

    
    ames = data.copy()
    
    ames.columns = ames.columns.str.lower().str.replace(' ', '_')
    ames.rename(columns={'id':'Id', 'saleprice':'SalePrice', 'year_remod/add':'year_remod', '2nd_flr_sf':'second_flr_sf', 
                         '1st_flr_sf':'first_flr_sf'}, inplace=True)

    
    ames['have_pool'] = np.where(ames['pool_qc'].isna(), 'No', 'Yes')
    ames['have_misc_features'] = np.where(ames['misc_feature'].isna(), 'No', 'Yes')
    ames['have_fence'] = np.where(ames['fence'].isna(), 'No', 'Yes')

    ames['porch_sf'] = ames['open_porch_sf'] + ames['enclosed_porch'] +ames['3ssn_porch'] + ames['screen_porch']

    for col in ['garage_area', 'total_bsmt_sf', 'bsmtfin_sf_1', 'bsmtfin_sf_2', 'mas_vnr_area', 'fireplaces', 'wood_deck_sf', 'porch_sf']:
        ames[f'have_{col}'] = np.where(ames[col]==0, 'No', 'Yes') 
        
            

    for val in [1, 2, 3]:
        ames[f'garage_cars_{val}'] = np.where(ames['garage_cars']==val, 'Yes', 'No') 

    ames['bsmtfin_type_2_unf'] = np.where(ames['bsmtfin_type_2'] == 'unf', 'Yes', 'No')


    for col in ['alley', 'fence', 'fireplace_qu', 'garage_finish', 'garage_type', 'bsmt_exposure', 'bsmt_cond', 'bsmt_qual', 
                    'mas_vnr_type', 'bsmtfin_type_1', 'bsmtfin_type_2', 'electrical', 'garage_cond', 'garage_qual']:
            ames[col].fillna('missing', inplace=True)
            
    number_words = {
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine',
        10: 'ten'
        }
    
    ames['overall_qual'] = ames['overall_qual'].map(number_words)
    ames['overall_cond'] = ames['overall_cond'].map(number_words)

    
    ames['bsmt_total_bath'] =ames['bsmt_full_bath'] + 0.5 * ames['bsmt_half_bath'] 

    ames['total_bath'] = ames['full_bath'] + 0.5 * ames['half_bath']

    ames['age'] = ames['yr_sold'] - ames['year_built']
    ames['new_construction'] = np.where(ames['age'] < 5, 'Yes', 'No')

    ames['age_by_remodel'] = ames['yr_sold'] - ames['year_remod']

    ames['house_remodeled'] = np.where(ames['year_built'] == ames['year_remod'], 'No', 'Yes')
    
    ames['total_area_sf'] = ames['gr_liv_area'] + ames['total_bsmt_sf'] 
    
    ames['total_area_sf_sq'] = ames['total_area_sf'] * ames['total_area_sf'] 
    # ames['gr_liv_area_squared'] = ames['gr_liv_area'] * ames['gr_liv_area']
    ames['age_squared'] = ames['age'] * ames['age']
    ames['age_by_remodel_squared'] = ames['age_by_remodel'] * ames['age_by_remodel']
    # ames['totrms_abvgrd_squared'] = ames['totrms_abvgrd'] * ames['totrms_abvgrd']
    ames['garage_area_X_garage_cars'] = ames['garage_area'] * ames['garage_cars']
    # ames['gr_liv_area_X_total_bsmt_sf'] = ames['gr_liv_area'] * ames['total_bsmt_sf']
    ames['gr_liv_area_X_garage_cars'] = ames['gr_liv_area'] * ames['garage_cars']
    ames['gr_liv_area_X_total_bath'] = ames['gr_liv_area'] * ames['total_bath']
    ames['gr_liv_area_X_bedroom_abvgr'] = ames['gr_liv_area'] * ames['bedroom_abvgr']
    ames['bsmtfin_sf_1_squared'] = ames['bsmtfin_sf_1'] * ames['bsmtfin_sf_1']

    
    ames['overall_qual'] = ames['overall_qual'].astype('object')
    ames['overall_cond'] = ames['overall_cond'].astype('object')

    
    numeric_col = ames.select_dtypes(include=['int64', 'float64']).isna().sum()
    numeric_with_missing = numeric_col[numeric_col > 0].index

    for col in numeric_with_missing:
        if version == '0':
            ames[col].fillna(0, inplace=True)
        elif version == 'mean':
            ames[col].fillna(ames[col].mean(), inplace=True)
               
    data, categorical_variable_label_dict = create_infrequent_category(ames.select_dtypes(include='object'), 
                                                                       threshold=20, 
                                                                       category_labels_dict=cat_labels)
    
    ames = pd.concat([ames.select_dtypes(include=['int64', 'float64']), data], axis=1)

    ames.drop(
        columns=['pid', 'pool_qc', 'misc_feature', 'garage_qual', 'bsmtfin_type_2', 'garage_yr_blt', 'garage_cond',
                 'ms_subclass', 'street', 'utilities', 'condition_2', 'roof_matl', 'exterior_2nd', 'heating',
                 'pool_area', 'misc_val', 'year_built', 'year_remod', 'bsmt_full_bath', 
                 'bsmt_half_bath', 'full_bath', 'half_bath'], inplace=True)

    return ames, categorical_variable_label_dict
