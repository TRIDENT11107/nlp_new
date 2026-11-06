"""Local best-effort persistence for trained Keras artifacts and run summaries."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

STORE=Path(__file__).resolve().parent/'saved_models'

def dataset_fingerprint(transactions):
    cols=[c for c in ('date','kind','category','amount','description') if c in transactions]
    raw=transactions[cols].sort_values(cols[:4]).to_csv(index=False).encode('utf-8') if len(transactions) else b''
    import hashlib
    return hashlib.sha256(raw).hexdigest()

def save_result(name,result,fingerprint):
    STORE.mkdir(parents=True,exist_ok=True)
    result['model'].save(STORE/f'{name}.keras')
    joblib.dump(result.get('scaler'),STORE/f'{name}_scaler.joblib')
    keep=('window','future','mae','rmse','mape','baseline','baseline_predicted','actual','predicted','history','split','epochs','best_epoch','batch_size','parameters','input_shape','train_samples','validation_samples','test_samples','test_indices','threshold','errors','feature_names','latent_dimension','threshold_quantile','threshold_source','threshold_training_errors')
    meta={k:result[k] for k in keep if k in result}; meta.update({'fingerprint':fingerprint,'status':'Trained'})
    if name=='autoencoder' and 'data' in result: meta['data']=result['data'].to_dict(orient='records')
    def conv(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,(np.integer,)): return int(v)
        if isinstance(v,(np.floating,)): return float(v)
        if isinstance(v,(np.bool_,)): return bool(v)
        if isinstance(v,tuple): return list(v)
        raise TypeError(type(v).__name__)
    (STORE/f'{name}.json').write_text(json.dumps(meta,default=conv,indent=2),encoding='utf-8')

def load_result(name,fingerprint):
    model_path=STORE/f'{name}.keras'; meta_path=STORE/f'{name}.json'; scaler_path=STORE/f'{name}_scaler.joblib'
    if not (model_path.exists() and meta_path.exists() and scaler_path.exists()): return None
    try:
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('fingerprint')!=fingerprint: return None
        result={k:meta[k] for k in meta if k not in ('fingerprint','status','data')}
        for key in ('future','baseline_predicted','actual','predicted','errors','test_indices','threshold_training_errors'):
            if key in result: result[key]=np.asarray(result[key])
        result['history']=meta.get('history',{})
        result['model']=tf.keras.models.load_model(model_path)
        result['scaler']=joblib.load(scaler_path)
        if name=='autoencoder' and 'data' in meta:
            result['data']=pd.DataFrame(meta['data']); result['features']=pd.DataFrame()
        return result
    except (OSError,ValueError,KeyError,TypeError,ImportError): return None
