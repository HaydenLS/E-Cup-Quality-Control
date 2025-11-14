import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(df, type='train'):
    """
    Basic data cleaning: drop duplicates and handle missing values.
    
    return: df_num, df_text
    """
    # --- getting num and text features

    # Список текстовых фич
    text_features_list = ['brand_name', 'description', 'name_rus', 'CommercialTypeName4', 'ItemID']
    if type == 'train':
        text_features_list.append('resolution')
    # Список числовых
    num_features_list = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_num = df[num_features_list]
    df_text = df[text_features_list]

    # =====================================
    # === Работа с числовыми признаками ===
    # =====================================

    # Missing in rating
    # New feature
    user_interactions = ['rating_1_count', 'rating_2_count', 'rating_3_count', 'rating_4_count', 'rating_5_count',
                      'comments_published_count', 'photos_published_count', 'videos_published_count']
    df_num.loc[:, "is_rating_exists"] = np.where(df_num['rating_1_count'].isnull(), 0, 1)
    # Fillna
    df_num.loc[:, user_interactions] = df_num[user_interactions].fillna(0)

    # Mising in Total
    period = 90 # период который мы оставим
    # New feature
    df_num[f'has_full_data_{period}d'] = (df_num[f'GmvTotal{period}'].notna() & 
                                               df_num[f'OrderAcceptedCountTotal{period}'].notna()).astype(int)
    df_num[f'missing_orders_only_{period}d'] = (df_num[f'GmvTotal{period}'].notna() & 
                                                     df_num[f'OrderAcceptedCountTotal{period}'].isna()).astype(int)
    # Fillna
    total_columns = []
    total_columns.extend([
        f'GmvTotal{period}',
        f'ExemplarAcceptedCountTotal{period}',
        f'OrderAcceptedCountTotal{period}',
        f'ExemplarReturnedCountTotal{period}', 
        f'ExemplarReturnedValueTotal{period}'
    ])
    df_num[total_columns] = df_num[total_columns].fillna(0)

    # Missing in itemcount
    df_num.loc[:, "is_item_count"] = np.where(df_num['ItemVarietyCount'].isnull(), 0, 1)
    # fillna
    df_num.loc[:, ['ItemVarietyCount', 'ItemAvailableCount']] = df_num[['ItemVarietyCount', 'ItemAvailableCount']].fillna(0)


    # --- Delete correlation features
    df_num = df_num.drop(columns=['ItemAvailableCount', 'OrderAcceptedCountTotal90'], axis=1)  # дубликат ItemVarietyCount
    # Оставляем только один период (например, 90 дней - самый полный)
    periods_to_drop = ['7', '30']  # удаляем данные за 7 и 30 дней
    for period in periods_to_drop:
        cols_to_drop = [col for col in df_num.columns if period in col and '90' not in col]
        df_num = df_num.drop(cols_to_drop, axis=1)
    
    # --- New features
    df_num['return_rate_90d'] = df_num['item_count_returns90'] / (df_num['item_count_sales90'] + 1)
    df_num['fake_return_rate_90d'] = df_num['item_count_fake_returns90'] / (df_num['item_count_sales90'] + 1)
    df_num['avg_order_value_90d'] = df_num['GmvTotal90'] / (df_num['ExemplarAcceptedCountTotal90'] + 1)

    # --- Delete seller id
    df_num.drop(columns=['SellerID'], inplace=True)


    



    # =====================================
    # === Работа с текстовыми признаками ===
    # =====================================

    # Базовая очистка/заполнение пропусков
    def clear_text(df):
        for c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
        return df

    df_text = clear_text(df_text)


    return df_num, df_text



# === Групповые статистики ===
def compute_group_stats(train_df):
    stats = {}
    grouped = train_df.groupby("CommercialTypeName4")
    for col in ["PriceDiscounted", "desc_len"]:
        med = grouped[col].median()
        iqr = grouped[col].quantile(0.75) - grouped[col].quantile(0.25)
        stats[col] = {"med": med, "iqr": iqr}
    return stats

def apply_group_stats(df, stats):
    for col, d in stats.items():
        med = df["CommercialTypeName4"].map(d["med"]).fillna(df[col].median())
        iqr = df["CommercialTypeName4"].map(d["iqr"]).fillna(df[col].quantile(0.75)-df[col].quantile(0.25))
        df[f"{col}_z"] = (df[col] - med) / (iqr + 1e-6)
    return df

# == Длина описания ==
def get_desc_length(df, id_col="id"):
    """
    Возвращает DataFrame со столбцом длины описания.

    """
    desc_len = df.copy()
    desc_len["desc_len"] = desc_len['description'].str.len()
    return desc_len['desc_len']

def df_extend(df_full, df_num, df_text, embeddings, emb_img, df_type: str ='train', group_stats=None):
    # Разделим данные

    # Возьмем категориальные признаки
    cat_cols = ["brand_name", "CommercialTypeName4"]
    cat_data = df_text[cat_cols].astype(str)

    # Добавим длину описания
    desc_len_df = get_desc_length(df_text, id_col="id")
    df_num = df_num.merge(desc_len_df, on="id", how="left")
    df_full = df_full.merge(desc_len_df, on="id", how="left")

    if df_type == 'train':
        # ====== Train/Val Data ======
        # 1) Трейн/тест сплит
        y = df_num["resolution"].astype(int).values
        data_train_num, data_val_num, y_train, y_val, train_cat, val_cat, embeddings_train, embeddings_val, emb_img_train, emb_img_val \
        = train_test_split(
            df_num, y, cat_data, embeddings, emb_img, test_size=0.21, stratify=y, random_state=41
        )

        # 2) Формируем полные датафреймы
        data_train_full = data_train_num.merge(train_cat, on='id', how='left')
        data_val_full = data_val_num.merge(val_cat, on='id', how='left')

        # 3) Получаем групповые статистики (z оценки и прочее)
        group_stats = compute_group_stats(data_train_full)
        
        # 4) Добавляем групповые статистики к данным
        data_train_full = apply_group_stats(data_train_full, group_stats)
        data_val_full = apply_group_stats(data_val_full, group_stats)
        
        # 5) После всего мы создаем числовые датафреймы
        data_train_num = data_train_full.drop(columns=cat_cols)
        data_val_num = data_val_full.drop(columns=cat_cols)

        return data_train_num, data_val_num, y_train, y_val, \
                train_cat, val_cat, embeddings_train, embeddings_val, emb_img_train, emb_img_val, group_stats
    
    elif df_type == 'test':
        assert group_stats is not None, "Для теста нужно передать обученные group_stats"

        # Применяем GroupStats
        data_test_full = df_num.merge(cat_data, on='id', how='left')
        df_num = apply_group_stats(data_test_full, group_stats)
        df_num.drop(columns=cat_cols, inplace=True)

        return df_num, cat_data


def normalize_seller_features(df, group_col='SellerID', features=None):
    """
    Нормализует признаки продавцов, беря максимальные значения для каждого продавца
    
    Parameters:
    df - DataFrame с данными
    group_col - колонка для группировки (SellerID)
    features - список признаков для нормализации
    """
    if features is None:
        features = ['seller_time_alive', 'GmvTotal90', 'ExemplarAcceptedCountTotal90', 
                   'ExemplarReturnedCountTotal90', 'ExemplarReturnedValueTotal90']
    
    # Создаем копию датафрейма
    result_df = df.copy()
    
    # Для каждого признака находим максимальное значение по SellerID
    for feature in features:
        max_values = df.groupby(group_col)[feature].transform('max')
        result_df[feature] = max_values
    
    return result_df

def prepare_seller_features(train_df, test_df, group_col='SellerID', features=None):
    """
    Подготавливает признаки продавцов для train и test, используя train данные
    для общих продавцов и test данные для новых продавцов
    """
    if features is None:
        features = ['seller_time_alive', 'GmvTotal90', 'ExemplarAcceptedCountTotal90', 
                   'ExemplarReturnedCountTotal90', 'ExemplarReturnedValueTotal90']
    
    # Нормализуем train данные
    train_normalized = normalize_seller_features(train_df, group_col, features)
    
    # Для test данных используем два подхода:
    test_normalized = test_df.copy()
    
    # Находим общих продавцов между train и test
    common_sellers = set(train_df[group_col].unique()) & set(test_df[group_col].unique())
    new_sellers = set(test_df[group_col].unique()) - set(train_df[group_col].unique())
    
    print(f"Общих продавцов: {len(common_sellers)}")
    print(f"Новых продавцов в test: {len(new_sellers)}")
    
    # Для общих продавцов берем значения из train (максимальные исторические значения)
    seller_max_values = train_df.groupby(group_col)[features].max()
    
    for feature in features:
        # Для общих продавцов используем значения из train
        mask_common = test_normalized[group_col].isin(common_sellers)
        test_normalized.loc[mask_common, feature] = test_normalized.loc[mask_common, group_col].map(seller_max_values[feature])
        
        # Для новых продавцов берем максимальные значения из test
        mask_new = test_normalized[group_col].isin(new_sellers)
        if mask_new.any():
            new_seller_max = test_df.groupby(group_col)[feature].max()
            test_normalized.loc[mask_new, feature] = test_normalized.loc[mask_new, group_col].map(new_seller_max)
    
    return train_normalized, test_normalized

def has_img_col(df, img_dir):
    # Создаем словарь: ItemID -> has_img
    img_dict = {}
    unique_ids = df['ItemID'].unique()
    
    for img_id in unique_ids:
        path = img_dir / f"{int(img_id)}.png"
        img_dict[img_id] = 1 if path.exists() else 0
    
    # Создаем копию и добавляем колонку через map
    new_df = df.copy()
    new_df['has_img'] = df['ItemID'].map(img_dict)
    
    return new_df

def load_text_embeddings(dir='text_emb'):
    """
    
    retrun: (embeddings, embeddings_test)
    """
    embeddings = np.load(f"{dir}/text_embeddings.npy")
    embeddings_test = np.load(f"{dir}/text_embeddings_test.npy")
    
    return embeddings, embeddings_test


    
def load_img_embeddings(dir='img_emb'):
    """
    retrun: (embeddings, embeddings_test)
    """
    train_img_emb = pd.read_parquet(f"{dir}/train_clip_img_emb.parquet")
    test_img_emb = pd.read_parquet(f"{dir}/test_clip_img_emb.parquet")
    
    return train_img_emb, test_img_emb
    

def load_pcas(dir='pca'):
    """
    retrun:
    """
    with open(f'{dir}/pca_text.pkl', 'rb') as le_dump_file:
        pca_text = pickle.load(le_dump_file)
    with open(f'{dir}/pca_image.pkl', 'rb') as le_dump_file:
        pca_image = pickle.load(le_dump_file)

    return pca_text, pca_image