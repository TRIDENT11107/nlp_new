import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import logging
from database import init_db,seed_demo,frame,execute
from deep_learning.models import expense_series,train_lstm,train_autoencoder
from deep_learning.preprocessing import monthly_financial_dataset,expense_windows,chronological_indices,lstm_dataset_preview,autoencoder_features,autoencoder_train_count,describe_dataset,DATASET_NAME,DATASET_SOURCE
from deep_learning.model_store import dataset_fingerprint,save_result,load_result
from pathlib import Path
from agents import decision_coordinator
st.set_page_config(page_title='InvestWise AI',layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--navy:#0B1220;--sidebar:#0D1728;--panel:#111C2E;--card:#172438;--line:#27364B;--emerald:#10B981;--text:#F8FAFC;--muted:#94A3B8;--warning:#F59E0B;--error:#EF4444}
html,body,[class*="css"],.stApp{font-family:Inter,system-ui,-apple-system,sans-serif!important;color:var(--text)}
.stApp{background:var(--navy)}
.block-container{max-width:1540px;padding:1.35rem 2.1rem 3rem}
[data-testid="stSidebar"]{background:var(--sidebar);border-right:1px solid var(--line)}
[data-testid="stSidebar"]>div:first-child{padding-top:1.1rem}
[data-testid="stSidebar"] [data-testid="stRadio"]>label{color:#64748b;font-size:.7rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin:.35rem 0 .6rem}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]{gap:5px}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]>label{padding:.57rem .72rem;border:1px solid transparent;border-radius:9px;color:#aab8cb;transition:background .15s,border-color .15s}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]>label:hover{background:#142338;border-color:#25364a;color:#fff}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]>label:has(input:checked){background:#12372f;border-color:#17664f;color:#e7fff5;font-weight:650}
h1,h2,h3{font-family:Inter,system-ui,sans-serif;letter-spacing:-.035em;color:var(--text)}h1{font-size:1.85rem!important;font-weight:700!important}h2{font-size:1.22rem!important;font-weight:650!important}h3{font-size:1rem!important}
p,li{color:#cbd5e1}small,.stCaption{color:var(--muted)!important}
.eyebrow{color:var(--emerald);letter-spacing:.14em;font-size:.68rem;font-weight:750}
.metric{height:100%;min-height:105px;background:linear-gradient(145deg,#19283c,#152236);border:1px solid var(--line);border-radius:13px;padding:15px 17px;box-shadow:0 8px 22px #02071120;transition:transform .16s,border-color .16s}
.metric:hover{border-color:#28765f;transform:translateY(-2px)}.label{font-size:.67rem;color:var(--muted);letter-spacing:.09em;font-weight:700}.value{font:650 1.36rem Inter,sans-serif;color:var(--text);margin-top:9px;letter-spacing:-.03em}.note{font-size:.7rem;color:#73d9b3;margin-top:4px}
.architecture{margin:.6rem 0 1.1rem;padding:1rem 1.1rem;border:1px solid var(--line);border-left:3px solid var(--emerald);border-radius:10px;background:linear-gradient(90deg,#14293a,#111c2e);line-height:1.8;color:#e2e8f0}.architecture b{font-size:.66rem;letter-spacing:.1em;color:#34d399}.architecture span{font-size:.82rem;color:var(--muted)}
.demo-badge{display:inline-flex;align-items:center;gap:9px;margin:.1rem 0 .9rem;padding:7px 11px;border:1px solid #315b4e;border-radius:999px;background:#10271f;color:#b7d8ca;font-size:.75rem}.demo-badge span{color:#34d399;font-size:.64rem;font-weight:800;letter-spacing:.09em}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:11px;overflow:hidden;background:#111c2e}
.stButton>button,.stDownloadButton>button{min-height:2.45rem;border:1px solid #33465d;border-radius:9px;background:#17283d;color:#e8f5f1;font-weight:600;transition:all .15s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--emerald);background:#12372f;color:#fff}
[data-testid="stForm"]{padding:1rem;background:#111c2e;border:1px solid var(--line);border-radius:13px}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:11px;background:#101b2b}
[data-testid="stMetric"]{background:#152338;border:1px solid var(--line);padding:13px 15px;border-radius:11px}
[data-testid="stTabs"] button[role="tab"]{font-weight:600;color:#94a3b8}
[data-testid="stTabs"] button[aria-selected="true"]{color:#eafff7;border-bottom-color:var(--emerald)}
[data-testid="stAlert"]{border-radius:10px}
input,textarea,[data-baseweb="select"]>div{border-radius:8px!important}
hr{border-color:var(--line)!important}
@media(max-width:800px){.block-container{padding:1rem .8rem 2rem}.metric{padding:13px;min-height:94px}.value{font-size:1.16rem}h1{font-size:1.55rem!important}}
</style>''',unsafe_allow_html=True)
try:
 init_db(); seed_demo()
except sqlite3.Error:
 st.error('The local financial database could not be initialized. Check available storage and restart the application.')
 st.stop()
def cash(x): return f'\u20b9{x:,.0f}'
def metric(k,v,n=''):
 st.markdown(f'<div class="metric"><div class="label">{k}</div><div class="value">{v}</div><div class="note">{n}</div></div>',unsafe_allow_html=True)
def getdata(): return frame('SELECT * FROM transactions ORDER BY date DESC,id DESC'),frame('SELECT * FROM budgets'),frame('SELECT * FROM investments')
def dataset_table(data,key,page_size=20):
 """Searchable, paginated dataset preview; Streamlit headers remain sortable."""
 if data.empty:
  st.info('No rows to display.'); return
 query=st.text_input('Search rows',key=f'{key}_search',placeholder='Search across displayed columns')
 view=data
 if query:
  mask=view.astype(str).apply(lambda col:col.str.contains(query,case=False,na=False,regex=False)).any(axis=1)
  view=view[mask]
 pages=max(1,(len(view)+page_size-1)//page_size)
 page=st.number_input('Page',min_value=1,max_value=pages,value=1,key=f'{key}_page')
 start=(int(page)-1)*page_size
 st.caption(f'{len(view):,} matching rows · page {int(page)} of {pages}; sort by selecting a column header.')
 st.dataframe(view.iloc[start:start+page_size],use_container_width=True,hide_index=True)
t,b,i=getdata(); month=pd.Timestamp.today().strftime('%Y-%m'); mt=t[t.date.str.startswith(month)] if len(t) else t
income=mt[mt.kind=='Income'].amount.sum() if len(mt) else 0; expense=mt[mt.kind=='Expense'].amount.sum() if len(mt) else 0; savings=income-expense; balance=t[t.kind=='Income'].amount.sum()-t[t.kind=='Expense'].amount.sum() if len(t) else 0; value=(i.quantity*i.current_price).sum() if len(i) else 0; cost=(i.quantity*i.purchase_price).sum() if len(i) else 0
profile=frame('SELECT * FROM profile WHERE id=1').iloc[0].to_dict()
fingerprint=dataset_fingerprint(t)
if st.session_state.get('model_fingerprint')!=fingerprint:
 st.session_state.lstm=load_result('lstm_expense',fingerprint)
 st.session_state.ae=load_result('autoencoder',fingerprint)
 st.session_state.lstm_status='Trained' if st.session_state.lstm else 'Not trained'
 st.session_state.ae_status='Trained' if st.session_state.ae else 'Not trained'
 st.session_state.model_fingerprint=fingerprint
coord=decision_coordinator(t,b,i,profile,month,st.session_state.get('lstm'),st.session_state.get('ae'))
score=coord['health']['score']
pages=['Dashboard','Dataset Explorer','Transactions','Budget Planner','Investment Portfolio','AI Financial Insights','Deep Learning Lab','AI Financial Assistant','Reports','Settings']
with st.sidebar:
 nav_icons={'Dashboard':'\u25eb','Dataset Explorer':'\u25a6','Transactions':'\u2261','Budget Planner':'\u25a4','Investment Portfolio':'\u25cc','AI Financial Insights':'\u2733','Deep Learning Lab':'\u2318','AI Financial Assistant':'\u25c9','Reports':'\u25a7','Settings':'\u2699'}
 st.markdown('<div style="display:flex;align-items:center;gap:10px;margin:.25rem 0 .45rem"><span style="display:grid;place-items:center;width:34px;height:34px;border-radius:10px;background:#12372f;color:#34d399;font-weight:800">IW</span><span style="font-size:1rem;font-weight:750;letter-spacing:-.03em;color:#f8fafc">InvestWise <span style="color:#10b981">AI</span></span></div>',unsafe_allow_html=True); st.caption('Intelligent Financial Management. Explainable Investment Decisions.'); page=st.radio('Navigate',pages,format_func=lambda item:f"{nav_icons[item]}   {item}",label_visibility='collapsed'); st.divider(); st.caption('DEMO MODE | INR'); st.caption('Synthetic sample data'); st.markdown('---'); st.caption('LOCAL PROFILE'); st.markdown('**Academic Demo User**'); st.caption('SQLite | local workspace')
 with st.expander('Demo data controls'):
  st.caption('Reset replaces ledger records with the reproducible synthetic demo dataset.')
  if st.button('Reset demo data',use_container_width=True): seed_demo(reset=True); st.rerun()
if page!='Dashboard':
 st.markdown('<div class="eyebrow">PERSONAL FINANCIAL INTELLIGENCE</div>',unsafe_allow_html=True); st.title(page)
if page=='Dashboard':
 now=pd.Timestamp.now()
 greeting='Good morning' if now.hour<12 else ('Good afternoon' if now.hour<18 else 'Good evening')
 st.markdown(f'<div class="eyebrow">FINANCIAL OVERVIEW | {now.strftime("%A, %d %B %Y").upper()}</div>',unsafe_allow_html=True)
 st.markdown(f'## {greeting}, welcome to InvestWise AI')
 st.caption('A clear view of your cash flow, spending, and tracked investments.')
 st.markdown('<div class="demo-badge"><span>DEMO DATA</span> Synthetic records | Manually entered investment prices | No live market quotes</div>',unsafe_allow_html=True)
 cols=st.columns(3)
 cards=[('TOTAL BALANCE',cash(balance),'All recorded income less expenses'),('MONTHLY INCOME',cash(income),'This month'),('MONTHLY EXPENSES',cash(expense),'This month'),('MONTHLY SAVINGS',cash(savings),f'{savings/income*100:.1f}% savings rate' if income else 'No income'),('PORTFOLIO VALUE',cash(value),'Tracked holdings | manual prices'),('FINANCIAL HEALTH',f'{score}/100','Illustrative indicator')]
 for idx,(k,v,note) in enumerate(cards):
  with cols[idx%3]: metric(k,v,note)
 if len(t):
  monthly=monthly_financial_dataset(t); chart_monthly=monthly.rename(columns={'income':'Income','expenses':'Expense'}); left,right=st.columns([1.45,1]);
  with left: st.subheader('Income and expenses'); st.plotly_chart(px.bar(chart_monthly,x='month',y=['Income','Expense'],barmode='group',color_discrete_map={'Income':'#10B981','Expense':'#F59E0B'}).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=0,r=0,t=12,b=0)),use_container_width=True)
  with right: st.subheader('Spending mix'); cats=t[t.kind=='Expense'].groupby('category',as_index=False).amount.sum(); st.plotly_chart(px.pie(cats,names='category',values='amount',hole=.65,color_discrete_sequence=px.colors.qualitative.Safe).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
  left,right=st.columns(2)
  with left:
   st.subheader('Portfolio allocation')
   if len(i): st.plotly_chart(px.bar(i.assign(value=i.quantity*i.current_price),x='asset',y='value',color='asset_type',color_discrete_sequence=['#43d49c','#529ef5','#e4b95e','#b594ff']).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
  with right:
   st.subheader('Budget utilization'); bb=b[b.month==month].copy(); actual=t[(t.kind=='Expense')&(t.date.str.startswith(month))].groupby('category').amount.sum();
   if len(bb): bb['actual']=bb.category.map(actual).fillna(0); st.plotly_chart(px.bar(bb,x='category',y=['actual','amount'],barmode='group',color_discrete_sequence=['#43d49c','#31445b']).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
  left,right=st.columns([1.4,1])
  with left:
   st.subheader('Savings trend'); monthly['savings']=monthly.income-monthly.expenses; st.plotly_chart(px.area(monthly,x='month',y='savings',color_discrete_sequence=['#10B981']).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=0,r=0,t=8,b=0)),use_container_width=True)
   st.subheader('Recent transactions'); st.dataframe(t[['date','kind','category','amount','description']].head(6),use_container_width=True,hide_index=True)
  with right:
   st.subheader('Financial health'); gauge=go.Figure(go.Indicator(mode='gauge+number',value=score,number={'suffix':' / 100','font':{'color':'#F8FAFC','size':28}},gauge={'axis':{'range':[0,100],'tickcolor':'#94A3B8'},'bar':{'color':'#10B981'},'bgcolor':'#172438','bordercolor':'#29384d','steps':[{'range':[0,40],'color':'#263044'},{'range':[40,70],'color':'#223d3a'},{'range':[70,100],'color':'#174638'}]})); gauge.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',height=240,margin=dict(l=12,r=12,t=24,b=0)); st.plotly_chart(gauge,use_container_width=True); st.caption('Illustrative indicator · not a certified financial assessment')
   st.subheader('AI insights'); st.write(f"Largest spending category: **{coord['health']['spending']['largest_category'] or '—'}**.")
   if coord['health']['budget']['overruns']:
    for over in coord['health']['budget']['overruns'][:2]: st.warning(f"{over['category']} budget used {over['utilization']:.0%}")
   else: st.success('No category budget overruns detected this month.')
  st.info('Fictional synthetic demo records. Investment prices are manually entered example values, not live market quotes.')
 else: st.info('Add financial records to populate your dashboard. Seeded demo data can be restored from the sidebar.')
elif page=='Dataset Explorer':
 st.caption('Inspect the exact local ledger and derived feature matrices used by both neural network pipelines.')
 if t.empty: st.info('The transaction dataset is empty. Add records or restore the reproducible demo dataset to explore model inputs.')
 else:
  synthetic=t.description.fillna('').str.contains('synthetic|demo',case=False,regex=True).all()
  classification='Synthetic demonstration data' if synthetic else ('Mixed synthetic and user-entered data' if t.description.fillna('').str.contains('synthetic|demo',case=False,regex=True).any() else 'Local user-entered data')
  tabs=st.tabs(['Dataset overview','LSTM dataset','Autoencoder dataset','Downloads'])
  with tabs[0]:
   overview=describe_dataset(t,'date'); cols=st.columns(4)
   for col,label,val in zip(cols,['DATASET','SOURCE TYPE','ROWS','COLUMNS'],[DATASET_NAME,classification,str(overview['rows']),str(overview['columns'])]):
    with col: metric(label,val)
   st.markdown('**Source** · '+DATASET_SOURCE); st.markdown('**Date range** · '+(f"{overview['date_range'][0]:%Y-%m-%d} to {overview['date_range'][1]:%Y-%m-%d}" if overview['date_range'] else 'No valid dates'))
   st.markdown('**Feature names** · '+', '.join(overview['features']))
   c1,c2=st.columns(2)
   with c1:
    st.markdown('**Feature types**'); st.dataframe(pd.DataFrame({'feature':list(overview['dtypes']),'dtype':list(overview['dtypes'].values())}),hide_index=True,use_container_width=True)
    st.markdown('**Missing values**'); st.dataframe(pd.DataFrame({'feature':list(overview['missing']),'missing':list(overview['missing'].values())}),hide_index=True,use_container_width=True)
   with c2:
    st.metric('Duplicate rows',overview['duplicates']); st.markdown('**Numeric summary statistics**'); st.dataframe(overview['summary'],use_container_width=True)
   with st.expander('Raw dataset preview'): dataset_table(t,'raw_dataset')
  with tabs[1]:
   monthly=monthly_financial_dataset(t); st.subheader('Monthly financial observations'); st.write(f"{len(monthly)} monthly rows · target variable: monthly total expenses · inputs: previous month totals in a {min(6,max(2,len(monthly)//3)) if len(monthly)>=6 else 0}-month lookback window")
   if len(monthly): st.plotly_chart(px.line(monthly,x='month',y=['income','expenses'],markers=True,color_discrete_map={'income':'#10B981','expenses':'#F59E0B'}).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',yaxis_title='Amount (INR)'),use_container_width=True)
   if len(monthly)>=6:
    preview,split=lstm_dataset_preview(monthly); st.markdown('**Preprocessing** · Missing months filled with zero. Target expenses use MinMaxScaler fitted on training target months only. Sliding windows predict the next monthly expense total; split is chronological with train, validation, then held-out test target windows.')
    c=st.columns(4)
    for col,label,val in zip(c,['SEQUENCE LENGTH','TRAIN SAMPLES','VALIDATION SAMPLES','TEST SAMPLES'],[str(split['window']),str(split['train']),str(split['validation']),str(split['test'])]):
     with col: metric(label,val)
    ranges={part:(preview.loc[preview.split==part,'target_month'].min(),preview.loc[preview.split==part,'target_month'].max()) for part in ('train','validation','test')}
    st.caption('Target date ranges · '+ ' · '.join(f"{part.title()}: {dates[0]:%b %Y}–{dates[1]:%b %Y}" for part,dates in ranges.items()))
    st.markdown('**Processed sliding-window preview** · each input column is a historical month; target is the following month.')
    dataset_table(preview,'lstm_dataset')
   else: st.warning('At least six monthly observations are required to construct the LSTM samples.')
  with tabs[2]:
   ae_raw,ae_scaled,scaler,feature_names=autoencoder_features(t)
   if len(ae_raw):
    st.subheader('Transaction feature matrix'); st.write('The Autoencoder trains on expense transactions only. Inputs are log-transformed amount, day of month, weekday, and an ordinal category index. StandardScaler is fitted on the first 80% of chronologically ordered expense rows and then applied to later rows.')
    c=st.columns(4)
    for col,label,val in zip(c,['EXPENSE ROWS','TRAIN SAMPLES','VALIDATION SAMPLES','INPUT FEATURES'],[str(len(ae_raw)),str(autoencoder_train_count(len(ae_raw))),str(len(ae_raw)-autoencoder_train_count(len(ae_raw))),str(len(feature_names))]):
     with col: metric(label,val)
    left,right=st.columns(2)
    with left: st.plotly_chart(px.histogram(ae_raw,x='amount',nbins=24,title='Expense amount distribution',color_discrete_sequence=['#10B981']).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True); st.plotly_chart(px.bar(ae_raw.category.value_counts().rename_axis('category').reset_index(name='transactions'),x='category',y='transactions',title='Expenses by category',color_discrete_sequence=['#10B981']).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
    with right:
     corr=ae_raw[feature_names].corr(numeric_only=True); st.plotly_chart(px.imshow(corr,text_auto='.2f',color_continuous_scale='Tealgrn',title='Feature correlation matrix').update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
     st.markdown('**Missing-value analysis**'); st.dataframe(ae_raw[feature_names].isna().sum().rename('missing').rename_axis('feature').reset_index(),hide_index=True,use_container_width=True)
    processed=ae_raw[['date','category','amount']].join(ae_scaled)
    ae_cut=autoencoder_train_count(len(ae_raw)); ae_dates=pd.to_datetime(ae_raw.date,errors='coerce')
    st.caption(f"Training dates: {ae_dates.iloc[0]:%b %Y}–{ae_dates.iloc[ae_cut-1]:%b %Y} · validation dates: {ae_dates.iloc[ae_cut]:%b %Y}–{ae_dates.iloc[-1]:%b %Y}. Features with missing values: {int(ae_raw[feature_names].isna().sum().sum())}. Threshold: training reconstruction error 95th percentile; flags identify unusual records for review, not fraud.")
    st.markdown('**Feature summary statistics**'); st.dataframe(ae_raw[feature_names].describe().T,use_container_width=True)
    st.markdown('**Model-ready standardized features**'); dataset_table(processed,'autoencoder_dataset')
   else: st.info('No valid expense transactions are available for autoencoder preprocessing.')
  with tabs[3]:
   monthly=monthly_financial_dataset(t); ae_raw,ae_scaled,_,_=autoencoder_features(t); processed=ae_raw[['id','date','category','amount','amount_log1p','day_of_month','weekday','category_index']].join(ae_scaled) if len(ae_raw) else pd.DataFrame()
   st.download_button('Download raw transaction CSV',t.to_csv(index=False),'investwise_raw_transactions.csv','text/csv')
   st.download_button('Download monthly LSTM dataset CSV',monthly.to_csv(index=False),'investwise_monthly_lstm.csv','text/csv')
   st.download_button('Download Autoencoder processed features CSV',processed.to_csv(index=False),'investwise_autoencoder_features.csv','text/csv')
   try: doc=Path(__file__).with_name('DATASET_DOCUMENTATION.md').read_bytes()
   except OSError: doc=b'Dataset source: local SQLite records. Synthetic seeded records and user-entered rows are not external benchmark data.'
   st.download_button('Download dataset description',doc,'DATASET_DOCUMENTATION.md','text/markdown')
elif page=='Transactions':
 st.caption('Add, filter, edit, and remove records stored in SQLite.')
 with st.expander('Add transaction'):
  with st.form('add'):
   c=st.columns(5); kind=c[0].selectbox('Type',['Expense','Income']); category=c[1].selectbox('Category',['Food','Shopping','Transportation','Education','Entertainment','Healthcare','Rent','Utilities','Subscriptions','Other expenses'] if kind=='Expense' else ['Salary','Freelancing','Business','Investments','Other income']); amount=c[2].number_input('Amount ₹',min_value=1.,value=1000.); date=c[3].date_input('Date'); desc=c[4].text_input('Description'); submit=st.form_submit_button('Save')
   if submit: execute('INSERT INTO transactions(date,kind,category,amount,description) VALUES(?,?,?,?,?)',(str(date),kind,category,amount,desc)); st.rerun()
 if len(t):
  c=st.columns(3); kf=c[0].selectbox('Type',['All','Income','Expense']); cf=c[1].selectbox('Category',['All']+sorted(t.category.unique())); q=c[2].text_input('Search'); view=t.copy()
  if kf!='All': view=view[view.kind==kf]
  if cf!='All': view=view[view.category==cf]
  if q: view=view[view.description.str.contains(q,case=False,na=False)]
  st.dataframe(view[['id','date','kind','category','amount','description']],use_container_width=True,hide_index=True)
  with st.expander('Edit or delete'):
   tid=st.selectbox('Record ID',t.id); row=t[t.id==tid].iloc[0]
   with st.form('edit'):
    c=st.columns(5); dt=c[0].date_input('Date',pd.to_datetime(row.date).date()); kd=c[1].selectbox('Type',['Income','Expense'],index=['Income','Expense'].index(row.kind)); cat=c[2].text_input('Category',row.category); amt=c[3].number_input('Amount',min_value=1.,value=float(row.amount)); de=c[4].text_input('Description',row.description); save=st.form_submit_button('Update'); delete=st.form_submit_button('Delete')
    if save: execute('UPDATE transactions SET date=?,kind=?,category=?,amount=?,description=? WHERE id=?',(str(dt),kd,cat,amt,de,int(tid))); st.rerun()
    if delete: execute('DELETE FROM transactions WHERE id=?',(int(tid),)); st.rerun()
elif page=='Budget Planner':
 st.caption('Set category-specific monthly limits and compare spending against plan.')
 with st.form('budget'):
  c=st.columns(3); mo=c[0].date_input('Month',pd.Timestamp.today().date().replace(day=1)); cat=c[1].selectbox('Category',['Food','Shopping','Transportation','Education','Entertainment','Healthcare','Rent','Utilities','Subscriptions']); amt=c[2].number_input('Limit ₹',min_value=1.,value=10000.); save=st.form_submit_button('Save budget')
  if save: execute('INSERT INTO budgets(month,category,amount) VALUES(?,?,?) ON CONFLICT(month,category) DO UPDATE SET amount=excluded.amount',(mo.strftime('%Y-%m'),cat,amt)); st.rerun()
bb=b[b.month==month].copy(); actual=t[(t.kind=='Expense')&(t.date.str.startswith(month))].groupby('category').amount.sum() if len(t) else pd.Series(dtype=float)
if page=='Budget Planner':
 if len(bb):
  bb['spent']=bb.category.map(actual).fillna(0); bb['remaining']=bb.amount-bb.spent
  for _,r in bb.iterrows(): st.markdown(f"**{r.category}** · {cash(r.spent)} / {cash(r.amount)} · remaining {cash(r.remaining)}"); st.progress(float(min(1,r.spent/r.amount)),text=f"{r.spent/r.amount:.0%} used"+(' · over budget' if r.spent>r.amount else ''))
  st.plotly_chart(px.bar(bb,x='category',y=['amount','spent'],barmode='group',color_discrete_sequence=['#344b63','#43d49c']).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
  selected=st.selectbox('Budget to remove',bb.id,key='budget_remove')
  if st.button('Delete budget'): execute('DELETE FROM budgets WHERE id=?',(int(selected),)); st.rerun()
 else: st.info('No budgets set for this month.')
elif page=='Investment Portfolio':
 st.caption('Holdings and returns based on your manually entered prices.')
 with st.form('inv'):
  c=st.columns(6); asset=c[0].text_input('Asset'); typ=c[1].selectbox('Type',['Stocks','Mutual funds','ETFs','Gold','Fixed deposits','Other assets']); qty=c[2].number_input('Quantity',min_value=.001,value=1.); buy=c[3].number_input('Purchase ₹',min_value=.01,value=100.); cur=c[4].number_input('Current ₹',min_value=.01,value=105.); dt=c[5].date_input('Purchase date'); save=st.form_submit_button('Add holding')
  if save and asset: execute('INSERT INTO investments(asset,asset_type,quantity,purchase_price,current_price,purchase_date) VALUES(?,?,?,?,?,?)',(asset,typ,qty,buy,cur,str(dt))); st.rerun()
 if len(i):
  i['invested']=i.quantity*i.purchase_price; i['value']=i.quantity*i.current_price; i['profit_loss']=i.value-i.invested; i['return_pct']=100*i.profit_loss/i.invested
  c=st.columns(3)
  with c[0]: metric('PORTFOLIO VALUE',cash(i.value.sum()))
  with c[1]: metric('INVESTED',cash(i.invested.sum()))
  with c[2]: metric('TOTAL RETURNS',cash(i.profit_loss.sum()),f"{i.profit_loss.sum()/i.invested.sum():.1%}")
  st.dataframe(i[['asset','asset_type','quantity','purchase_price','current_price','invested','value','profit_loss','return_pct']],use_container_width=True,hide_index=True); st.plotly_chart(px.pie(i,names='asset_type',values='value',hole=.5).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
  with st.expander('Edit a holding'):
   aid=st.selectbox('Holding',i.id,key='investment_edit'); row=i[i.id==aid].iloc[0]
   with st.form('edit_investment'):
    types=['Stocks','Mutual funds','Mutual fund','ETFs','ETF','Gold','Fixed deposits','Fixed deposit','Other assets']; idx=types.index(row.asset_type) if row.asset_type in types else len(types)-1
    c=st.columns(5); ename=c[0].text_input('Asset name',row.asset); etype=c[1].selectbox('Asset type',types,index=idx); eqty=c[2].number_input('Quantity',min_value=.001,value=float(row.quantity)); ebuy=c[3].number_input('Purchase price',min_value=.01,value=float(row.purchase_price)); ecur=c[4].number_input('Current price',min_value=.01,value=float(row.current_price)); update=st.form_submit_button('Update holding')
    if update: execute('UPDATE investments SET asset=?,asset_type=?,quantity=?,purchase_price=?,current_price=? WHERE id=?',(ename,etype,eqty,ebuy,ecur,int(aid))); st.rerun()
   if st.button('Delete selected holding',key='delete_investment'): execute('DELETE FROM investments WHERE id=?',(int(aid),)); st.rerun()
elif page=='AI Financial Insights':
 st.caption('Explainable summaries computed from the saved local records.')
 if len(t):
  report=coord['health']; st.subheader('Spending agent'); st.write(f"Largest expense category: **{report['spending']['largest_category'] or '—'}**, totaling **{cash(report['spending']['largest_total'])}**.")
  st.subheader('Savings agent'); st.write(f"This month savings: **{cash(report['savings']['net_savings'])}** ({report['savings']['savings_rate']:.1%} of income)." if report['savings']['income'] else 'No income recorded this month.')
  st.subheader('Decision Coordinator'); st.caption('Consolidates measured spending, budget, portfolio, health, and any trained model outputs.')
  for insight in coord['insights']:
   if insight['priority']=='high': st.warning(insight['text'])
   else: st.info(insight['text'])
   with st.expander('Evidence, reasoning, and limitations for each finding'):
    for insight in coord['insights']:
     st.markdown(f"**{insight['source'].title()} · {insight['text']}**")
     st.caption(f"Supporting data: {insight.get('supporting_data') or 'No matching records.'} · Reasoning: {insight.get('reasoning','Calculated from saved records.')} · Suggested review: {insight.get('recommended_action','Review the source records.')} · Limitation: {insight.get('limitation','Based only on local records.')}")
   st.subheader('Agent status')
   st.write('Market · Risk · Portfolio · Spending · Savings · Learning · Explanation · Notification')
   st.caption(f"Tracked prices are manual (live data: {coord['agents']['market']['live_data']}); {len(coord['agents']['notification']['items'])} budget or unusual-transaction notification(s) are available for review.")
  st.subheader(f'Illustrative financial health indicator · {score}/100'); st.write('Calculated from current records: savings rate 30%, expense burden 25%, budget adherence 20%, profile emergency reserve 15%, and portfolio diversification 10%. This is not a certified assessment or individualized investment advice.')
 else: st.info('Record transactions to generate insights.')
elif page=='Deep Learning Lab':
 st.caption('Neural network monitoring and evaluation. Models train only when you request it and are cached against the current transaction dataset.')
 lstm_tab,ae_tab,notes_tab=st.tabs(['LSTM expense forecaster','Autoencoder anomaly detector','Model notes'])
 with lstm_tab:
  r=st.session_state.get('lstm'); status=st.session_state.get('lstm_status','Not trained')
  st.subheader('LSTM expense forecasting')
  st.caption('Chronological monthly holdout ? neural forecast compared with a last-value baseline')
  st.markdown('<div class="architecture"><b>NETWORK ARCHITECTURE</b><br>Input window &rarr; LSTM 64 &rarr; Dropout 0.20 &rarr; LSTM 32 &rarr; Dense 16 ReLU &rarr; Expense prediction<br><span>Adam optimizer | Mean squared error | Early stopping on validation loss</span></div>',unsafe_allow_html=True)
  if st.button('Train / retrain LSTM',key='train_lstm'):
   st.session_state.lstm_status='Training'
   try:
    with st.spinner('Training LSTM and evaluating chronological holdout…'):
     trained=train_lstm(expense_series(t)); save_result('lstm_expense',trained,fingerprint); st.session_state.lstm=trained; st.session_state.lstm_status='Trained'; st.session_state.model_fingerprint=fingerprint
    st.success('LSTM training and holdout evaluation completed.')
   except Exception: logging.exception('LSTM training failed'); st.session_state.lstm_status='Training failed'; st.error('LSTM training failed. Check the app logs and verify that enough monthly observations are available.')
   st.rerun()
  r=st.session_state.get('lstm')
  if r:
   cards=st.columns(6); vals=[status,cash(r['mae']),cash(r['rmse']),f"{r['mape']:.2f}%" if r.get('mape') is not None else 'N/A',str(r['parameters']),str(r['test_samples'])]
   for col,label,val in zip(cards,['MODEL STATUS','TEST MAE','TEST RMSE','TEST MAPE','PARAMETERS','TEST WINDOWS'],vals):
    with col: metric(label,val)
   st.markdown(f"Input shape: `{r['input_shape']}` · sequence length `{r['window']}` · optimizer Adam · loss MSE · maximum epochs `{r['epochs']}` · early stopping patience 8 · batch size `{r['batch_size']}` · train/validation/test `{r['train_samples']}/{r['validation_samples']}/{r['test_samples']}`.")
   st.subheader('Training and validation loss'); st.plotly_chart(px.line(pd.DataFrame({'epoch':np.arange(1,len(r['history']['loss'])+1),'Training loss':r['history']['loss'],'Validation loss':r['history']['val_loss']}),x='epoch',y=['Training loss','Validation loss'],markers=True,color_discrete_sequence=['#10B981','#F59E0B']).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
   monthly=monthly_financial_dataset(t); labels=monthly.month.iloc[r['test_indices']].dt.strftime('%b %Y').tolist(); hist=pd.DataFrame({'month':labels,'Actual':r['actual'],'LSTM':r['predicted'],'Naive last value':r['baseline_predicted']}); st.subheader('Held-out test: actual vs predicted'); st.plotly_chart(px.line(hist,x='month',y=['Actual','LSTM','Naive last value'],markers=True,color_discrete_sequence=['#F8FAFC','#10B981','#F59E0B']).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
   resid=pd.DataFrame({'month':labels,'residual':r['actual']-r['predicted']}); st.subheader('Prediction residuals · actual minus forecast'); st.plotly_chart(px.bar(resid,x='month',y='residual',color='residual',color_continuous_scale=['#EF4444','#10B981']).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
   base=r['baseline']; st.markdown(f"**Naive baseline comparison** · MAE {cash(base['mae'])}, RMSE {cash(base['rmse'])}. LSTM MAE is {'lower' if r['mae']<base['mae'] else 'not lower'} than this held-out baseline; no superiority is assumed.")
   future_dates=pd.date_range(pd.Timestamp.today().replace(day=1)+pd.offsets.MonthBegin(1),periods=len(r['future']),freq='MS'); st.subheader('Recursive expense forecast'); st.plotly_chart(px.line(pd.DataFrame({'month':future_dates,'forecast':r['future']}),x='month',y='forecast',markers=True).update_traces(line_color='#10B981').update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',yaxis_title='INR'),use_container_width=True)
  else: st.info('No trained LSTM result for this dataset yet. Train the model to calculate real metrics.')
 with ae_tab:
  ae=st.session_state.get('ae'); status=st.session_state.get('ae_status','Not trained')
  st.subheader('Autoencoder anomaly detection')
  st.caption('Chronological transaction rows | reconstruction error highlights records for review')
  st.markdown('<div class="architecture"><b>NETWORK ARCHITECTURE</b><br>4 standardized inputs &rarr; Dense 8 &rarr; Latent 3 &rarr; Dense 8 &rarr; 4-value reconstruction<br><span>Adam optimizer | Mean squared error | Threshold from training reconstruction errors</span></div>',unsafe_allow_html=True)
  if st.button('Train / retrain autoencoder',key='train_ae'):
   st.session_state.ae_status='Training'
   try:
    with st.spinner('Training autoencoder and scoring expense transactions…'):
     trained=train_autoencoder(t); save_result('autoencoder',trained,fingerprint); st.session_state.ae=trained; st.session_state.ae_status='Trained'; st.session_state.model_fingerprint=fingerprint
    st.success('Autoencoder training completed.')
   except Exception: logging.exception('Autoencoder training failed'); st.session_state.ae_status='Training failed'; st.error('Autoencoder training failed. Check the app logs and verify that enough expense rows are available.')
   st.rerun()
  ae=st.session_state.get('ae')
  if ae:
   cards=st.columns(6)
   for col,label,val in zip(cards,['MODEL STATUS','TRAIN ROWS','VALIDATION ROWS','ERROR THRESHOLD','FLAGGED ROWS','PARAMETERS'],[status,str(ae['train_samples']),str(ae['validation_samples']),f"{ae['threshold']:.5f}",str(int(ae['data'].anomaly.sum())),str(ae['parameters'])]):
    with col: metric(label,val)
   st.caption(f"Input shape `{ae['input_shape']}` · Feature normalization: StandardScaler fitted on training rows · optimizer Adam · loss MSE · threshold: {ae['threshold_quantile']:.0%} empirical quantile of training reconstruction errors · maximum epochs {ae['epochs']} · early stopping patience 6 · batch size {ae['batch_size']}.")
   st.subheader('Training and validation loss'); st.plotly_chart(px.line(pd.DataFrame({'epoch':np.arange(1,len(ae['history']['loss'])+1),'Training loss':ae['history']['loss'],'Validation loss':ae['history']['val_loss']}),x='epoch',y=['Training loss','Validation loss'],markers=True,color_discrete_sequence=['#10B981','#F59E0B']).update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
   err=pd.DataFrame({'reconstruction error':ae['errors'],'status':np.where(ae['errors']>ae['threshold'],'Flagged','Typical')}); fig=px.histogram(err,x='reconstruction error',color='status',nbins=30,color_discrete_map={'Typical':'#10B981','Flagged':'#EF4444'},title='Reconstruction error distribution'); fig.add_vline(x=ae['threshold'],line_dash='dash',line_color='#F59E0B',annotation_text='95th percentile threshold'); st.plotly_chart(fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
   st.subheader('Transaction reconstruction scores'); st.plotly_chart(px.scatter(ae['data'],x='date',y='amount',color=ae['data'].anomaly.map({True:'Flagged',False:'Typical'}),color_discrete_map={'Typical':'#10B981','Flagged':'#EF4444'},hover_data=['category','reconstruction_error'],title='Typical and flagged expense rows').update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)'),use_container_width=True)
   st.markdown('**Highest reconstruction errors**'); st.dataframe(ae['data'].sort_values('reconstruction_error',ascending=False)[['date','category','amount','description','reconstruction_error','anomaly']].head(10),use_container_width=True,hide_index=True)
  else: st.info('No trained Autoencoder result for this dataset yet. Train the model to generate reconstruction errors and a data-derived threshold.')
 with notes_tab:
  st.markdown('**Why LSTM?** Its recurrent gates learn relationships across ordered monthly observations. A sliding window supplies recent expenses; the final dense layer predicts the next month. The model is evaluated against later months held out by time, with a last-value naive baseline.')
  st.markdown('**Why an autoencoder?** It learns a compressed representation of expense feature rows and reconstructs them. Large reconstruction errors indicate rows the model represents less well; the 95th-percentile cutoff is a review flag, not a fraud label.')
  st.warning('Demo history is short and synthetic. Forecasts and flags are educational decision support, not dependable financial advice. Model artifacts are cached locally and tied to a fingerprint of the transaction dataset.')
elif page=='AI Financial Assistant':
 st.caption('Rule-based, local question answering against your SQLite records; no API key required.')
 q=st.chat_input('Ask about spending, savings, budgets, investments, anomalies, or forecast')
 if q:
  st.session_state.setdefault('chat',[]).append(('user',q)); low=q.lower(); ex=t[t.kind=='Expense']; sums=ex.groupby('category').amount.sum().sort_values(ascending=False)
  if any(x in low for x in ['spend','most','category']): ans=f'Largest spending category: {sums.index[0]} at {cash(sums.iloc[0])}.' if len(sums) else 'No expenses yet.'
  elif 'sav' in low: ans=f'Current month income {cash(income)}, expenses {cash(expense)}, net savings {cash(savings)}.'
  elif 'portfolio' in low or 'invest' in low: ans=f'Portfolio value {cash(value)} at manual/demo entered prices.'
  elif 'budget' in low:
   over=[f"{r.category}: {cash(actual.get(r.category,0))}/{cash(r.amount)}" for _,r in bb.iterrows() if actual.get(r.category,0)>r.amount]; ans='Over budget: '+', '.join(over) if over else 'No budget overruns recorded this month.'
  elif 'unusual' in low or 'anomal' in low: ans='Train the Autoencoder in Deep Learning Lab to detect high reconstruction errors.'
  elif 'forecast' in low or 'next month' in low: ans='Train the LSTM in Deep Learning Lab to generate an expense forecast.'
  elif 'health' in low: ans=f'Illustrative health indicator is {score}/100. See Insights for its weighted components; it is not a certified assessment.'
  elif 'highest expense' in low: ans=f'Largest single expense is {cash(ex.amount.max())}.' if len(ex) else 'No expenses recorded.'
  else: ans='I can answer questions about categories, monthly savings, portfolio value, budget overruns, anomaly analysis, forecasts, and financial health.'
  st.session_state.chat.append(('assistant',ans))
 for role,msg in st.session_state.get('chat',[]):
  with st.chat_message(role): st.write(msg)
elif page=='Reports':
 st.caption('CSV exports of transaction, monthly, and portfolio summaries.')
 if len(t):
  t['month']=t.date.str[:7]; sm=t.pivot_table(index='month',columns='kind',values='amount',aggfunc='sum',fill_value=0).reset_index(); sm['net_savings']=sm.get('Income',0)-sm.get('Expense',0); st.dataframe(sm,use_container_width=True,hide_index=True); st.download_button('Monthly summary CSV',sm.to_csv(index=False),'investwise_monthly.csv'); st.download_button('Transactions CSV',t.to_csv(index=False),'investwise_transactions.csv')
  if len(i): st.download_button('Portfolio CSV',i.to_csv(index=False),'investwise_portfolio.csv')
 st.caption('PDF export is not enabled in this local build.')
elif page=='Settings':
 st.caption('Local profile settings and database details.')
 profile=frame('SELECT * FROM profile WHERE id=1').iloc[0]
 with st.form('profile'):
  c=st.columns(2); inc0=c[0].number_input('Expected monthly income ₹',min_value=0.,value=float(profile.monthly_income)); reserve=c[1].number_input('Emergency savings ₹',min_value=0.,value=float(profile.emergency_savings)); save=st.form_submit_button('Save profile')
  if save: execute('UPDATE profile SET monthly_income=?,emergency_savings=? WHERE id=1',(inc0,reserve)); st.success('Saved.')
 st.write(f'{len(t)} transactions · {len(b)} budgets · {len(i)} holdings'); st.info('Records persist in local SQLite storage for this run environment. Hosted app storage can be ephemeral; reset or export data as needed.')
