"""Reproducible dataset construction shared by the explorer and model code."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATASET_NAME='InvestWise AI local transaction ledger'
DATASET_SOURCE='Local SQLite records (fictional seeded demo or user-entered rows)'

def monthly_financial_dataset(transactions):
    """Aggregate transaction data by month and fill missing months with zero."""
    cols=['month','income','expenses']
    if transactions.empty: return pd.DataFrame(columns=cols)
    d=transactions.copy(); d['date']=pd.to_datetime(d.date,errors='coerce'); d=d.dropna(subset=['date'])
    if d.empty: return pd.DataFrame(columns=cols)
    d['month']=d.date.dt.to_period('M').dt.to_timestamp()
    wide=d.pivot_table(index='month',columns='kind',values='amount',aggfunc='sum',fill_value=0)
    for kind in ('Income','Expense'):
        if kind not in wide: wide[kind]=0.0
    idx=pd.date_range(wide.index.min(),wide.index.max(),freq='MS')
    wide=wide.reindex(idx,fill_value=0).rename_axis('month')
    return pd.DataFrame({'month':wide.index,'income':wide['Income'].astype(float).values,'expenses':wide['Expense'].astype(float).values})

def expense_windows(monthly, window=None):
    """Build ordered one-step windows and their target month indices."""
    values=np.asarray(monthly['expenses'] if isinstance(monthly,pd.DataFrame) else monthly,dtype=float)
    if len(values)<6: raise ValueError('At least six monthly observations are needed for an LSTM.')
    w=int(window or min(6,max(2,len(values)//3)))
    if w<1 or w>=len(values): raise ValueError('Sequence length must be positive and shorter than the series.')
    X=np.stack([values[j-w:j] for j in range(w,len(values))]); y=values[w:]
    return X,y,w,np.arange(w,len(values))

def chronological_indices(sample_count):
    """Return disjoint increasing train/validation/test target index ranges."""
    if sample_count<3: raise ValueError('At least three sliding windows are required for chronological splitting.')
    train_end=max(1,int(sample_count*.65)); val_end=max(train_end+1,int(sample_count*.82)); val_end=min(val_end,sample_count-1)
    return {'train':np.arange(0,train_end),'validation':np.arange(train_end,val_end),'test':np.arange(val_end,sample_count),'train_end':train_end,'validation_end':val_end}

def lstm_dataset_preview(monthly):
    X,y,w,target_idx=expense_windows(monthly)
    parts=chronological_indices(len(y)); scaler=MinMaxScaler().fit(y[parts['train']].reshape(-1,1))
    rows=[]
    for j,(seq,target,index) in enumerate(zip(X,y,target_idx)):
        split='train' if j<parts['train_end'] else ('validation' if j<parts['validation_end'] else 'test')
        row={'target_month':pd.Timestamp(monthly.iloc[index].month),'actual_expenses':float(target),'scaled_target':float(scaler.transform([[target]])[0,0]),'split':split}
        row.update({f'input_t-{w-k}':float(v) for k,v in enumerate(seq)})
        rows.append(row)
    return pd.DataFrame(rows),{'window':w,'train':parts['train_end'],'validation':parts['validation_end']-parts['train_end'],'test':len(y)-parts['validation_end'],'train_boundary':pd.Timestamp(monthly.iloc[w+parts['train_end']-1].month),'validation_boundary':pd.Timestamp(monthly.iloc[w+parts['validation_end']-1].month),'test_start':pd.Timestamp(monthly.iloc[w+parts['validation_end']].month)}

def autoencoder_features(transactions):
    """Return raw and standardized expense features actually consumed by AE."""
    d=transactions[transactions.kind=='Expense'].copy()
    if d.empty: return d, pd.DataFrame(), None, []
    dates=pd.to_datetime(d.date,errors='coerce'); d=d.loc[dates.notna()].copy(); d['_parsed_date']=pd.to_datetime(d.date); d=d.sort_values(['_parsed_date','id'] if 'id' in d.columns else ['_parsed_date']).drop(columns='_parsed_date'); dates=pd.to_datetime(d.date)
    cats=sorted(d.category.fillna('Unknown').astype(str).unique()); category_index={x:j for j,x in enumerate(cats)}
    d['amount']=pd.to_numeric(d.amount,errors='coerce'); d=d.loc[d.amount.notna() & (d.amount>0)].copy(); dates=pd.to_datetime(d.date)
    d['amount_log1p']=np.log1p(d.amount.astype(float)); d['day_of_month']=dates.dt.day.astype(int); d['weekday']=dates.dt.dayofweek.astype(int); d['category_index']=d.category.fillna('Unknown').astype(str).map(category_index).astype(int)
    feature_names=['amount_log1p','day_of_month','weekday','category_index']
    raw=d[feature_names].copy(); scaler=StandardScaler(); train_end=autoencoder_train_count(len(raw)); scaler.fit(raw.iloc[:train_end]); scaled=pd.DataFrame(scaler.transform(raw),columns=[f'{x}_z' for x in feature_names],index=d.index)
    return d,scaled,scaler,feature_names

def autoencoder_train_count(n):
    return max(1,min(n-1,int(n*.8))) if n>1 else n

def describe_dataset(data, date_column=None):
    """Return calculated overview values for the currently loaded frame."""
    date_range=None
    if date_column and date_column in data:
        dates=pd.to_datetime(data[date_column],errors='coerce').dropna()
        if len(dates): date_range=(dates.min(),dates.max())
    return {'rows':len(data),'columns':len(data.columns),'features':list(data.columns),'dtypes':{c:str(t) for c,t in data.dtypes.items()},'missing':data.isna().sum().astype(int).to_dict(),'duplicates':int(data.duplicated().sum()),'date_range':date_range,'summary':data.select_dtypes(include='number').describe().T}
