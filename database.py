"""SQLite persistence and reproducible demo data for InvestWise AI."""
from pathlib import Path
import sqlite3
import pandas as pd
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "data" / "investwise.db"
DB_PATH.parent.mkdir(exist_ok=True)

def connect():
    c=sqlite3.connect(DB_PATH); c.row_factory=sqlite3.Row; return c

@contextmanager
def connection():
    c=connect()
    try:
        with c: yield c
    finally:
        c.close()

def init_db():
    with connection() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS transactions(id INTEGER PRIMARY KEY, date TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('Income','Expense')), category TEXT NOT NULL, amount REAL NOT NULL CHECK(amount>0), description TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS budgets(id INTEGER PRIMARY KEY, month TEXT NOT NULL, category TEXT NOT NULL, amount REAL NOT NULL CHECK(amount>0), UNIQUE(month,category));
        CREATE TABLE IF NOT EXISTS investments(id INTEGER PRIMARY KEY, asset TEXT NOT NULL, asset_type TEXT NOT NULL, quantity REAL NOT NULL, purchase_price REAL NOT NULL, current_price REAL NOT NULL, purchase_date TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS profile(id INTEGER PRIMARY KEY CHECK(id=1), monthly_income REAL DEFAULT 0, emergency_savings REAL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS asset_prices(id INTEGER PRIMARY KEY, asset TEXT, date TEXT, price REAL);
        CREATE TABLE IF NOT EXISTS notifications(id INTEGER PRIMARY KEY, date TEXT, message TEXT, level TEXT, read INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS forecasts(id INTEGER PRIMARY KEY, created_at TEXT, model TEXT, value REAL, details TEXT);
        CREATE TABLE IF NOT EXISTS anomalies(id INTEGER PRIMARY KEY, transaction_id INTEGER, error REAL, flagged INTEGER, created_at TEXT);
        ''')
        c.execute('INSERT OR IGNORE INTO profile(id) VALUES(1)')
        # Older seed runs could bind NumPy int64 values as SQLite blobs.
        # Normalize those amounts on startup so numeric aggregates stay valid.
        rows=c.execute("SELECT id,amount FROM transactions WHERE typeof(amount)='blob'").fetchall()
        for row in rows:
            raw=row['amount']
            if isinstance(raw,bytes) and len(raw) in (1,2,4,8):
                c.execute('UPDATE transactions SET amount=? WHERE id=?',(int.from_bytes(raw,'little',signed=True),row['id']))

def frame(sql, params=()):
    with connection() as c: return pd.read_sql_query(sql,c,params=params)

def execute(sql, params=()):
    with connection() as c: cur=c.execute(sql,params); return cur.lastrowid

def seed_demo(reset=False):
    import numpy as np
    from datetime import date
    with connection() as c:
        if reset:
            for t in ('transactions','budgets','investments','asset_prices','notifications','forecasts','anomalies'): c.execute(f'DELETE FROM {t}')
        elif c.execute('SELECT count(*) FROM transactions').fetchone()[0]: return
        rng=np.random.default_rng(27)
        months=pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1),periods=12,freq='MS')
        categories={'Food':(9000,2200),'Shopping':(6000,3200),'Transportation':(3500,900),'Education':(2800,500),'Entertainment':(2500,800),'Healthcare':(1800,1000),'Rent':(15000,0),'Utilities':(2200,500),'Subscriptions':(800,250)}
        for mi,m in enumerate(months):
            c.execute('INSERT INTO transactions(date,kind,category,amount,description) VALUES(?,?,?,?,?)',(m.strftime('%Y-%m-01'),'Income','Salary',75000+mi*500,'Monthly salary · synthetic demo'))
            c.execute('INSERT INTO transactions(date,kind,category,amount,description) VALUES(?,?,?,?,?)',(m.strftime('%Y-%m-18'),'Income','Freelancing',int(rng.integers(4000,12000)),'Project income · synthetic demo'))
            for cat,(base,sd) in categories.items():
                amt=max(200,float(rng.normal(base,sd)))
                c.execute('INSERT INTO transactions(date,kind,category,amount,description) VALUES(?,?,?,?,?)',(m.strftime('%Y-%m-%d'), 'Expense',cat,round(amt,2),'Synthetic demonstration record'))
                c.execute('INSERT OR IGNORE INTO budgets(month,category,amount) VALUES(?,?,?)',(m.strftime('%Y-%m'),cat,base*1.18))
        for asset,typ,q,buy,now in [('Nifty 50 Index','ETF',18,18500,22400),('Bluechip Equity','Mutual fund',42,510,584),('Gold ETF','Gold',30,52,68),('Govt bond','Fixed deposit',1,50000,53500)]:
            c.execute('INSERT INTO investments(asset,asset_type,quantity,purchase_price,current_price,purchase_date) VALUES(?,?,?,?,?,?)',(asset,typ,q,buy,now,'2025-01-15'))
            for d,p in zip(months,np.linspace(buy,now,len(months))*(1+rng.normal(0,.018,len(months)))): c.execute('INSERT INTO asset_prices(asset,date,price) VALUES(?,?,?)',(asset,d.strftime('%Y-%m-%d'),round(float(p),2)))
        c.execute('UPDATE profile SET monthly_income=?,emergency_savings=? WHERE id=1',(80000,150000))
