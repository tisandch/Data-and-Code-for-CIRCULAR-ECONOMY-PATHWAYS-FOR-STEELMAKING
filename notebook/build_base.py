"""Побудова відредагованої бази data/mbc_base.csv з таблиць OECD ICIO.

Запуск із папки проєкту: python build_base.py  (або по комірках # %%).
"""


# %% [markdown]
# # 01. Побудова відредагованої бази `mbc_base.csv` з таблиць OECD ICIO
#
# **Вхід:**
# * повні таблиці OECD ICIO 2025 `raw/icio_full/{YEAR}_SML.csv` (81 країна + ROW × 50 галузей) — **основне джерело**; зараз є 2022;
# * для років без повної таблиці — вивантаження `raw/icio/{COUNTRY}-{YEAR}.csv` (IND, SAU, UKR, USA × 2015, 2019, 2021).
#
# Коли з'являться повні таблиці за 2015, 2019, 2021 — достатньо покласти їх у `raw/icio_full/` і перезапустити блокнот.
# **Вихід:** `data/mbc_base.csv` — єдина база для розрахунків; `data/corrections_log.csv` — журнал усіх виправлень; `data/validation_report.csv` — перевірки після виправлень.
#
# Кожне виправлення виконується кодом і записується в журнал, тому результат повністю відтворюваний.


# %%
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path('raw/icio')
OUT = Path('data'); OUT.mkdir(exist_ok=True)
pd.set_option('display.width', 220, 'display.max_columns', 40)

# Агрегування кодів ICIO у 13 галузей і компоненти рахунку
ICIO_MAP = {
    'B05': 'COAL', 'B06': 'OILGAS', 'B07': 'ORES', 'C19': 'COKE', 'C24A': 'METAL', 'C25': 'FABMET',
    'D': 'ELEC', 'E': 'WASTE', 'F': 'CONSTR', 'G': 'TRADE',
    'H49': 'TRANSP', 'H50': 'TRANSP', 'H51': 'TRANSP', 'H52': 'WAREH', 'OTHERS': 'OTHERS',
    'TOTAL_IC': 'TOTAL_IC',
    'HFCE': 'HH', 'NPISH': 'NPISH', 'GGFC': 'GOV', 'GFCF': 'GFCF', 'INVNT': 'INVNT', 'DPABR': 'DPABR',
    'OUT': 'OUTPUT', 'EXP': 'EXPORT', 'IMP': 'IMPORT',
    'VA': None,   # у джерелі = OUT − TOTAL_IC за рядком, тобто не ВДВ → не переноситься
}
IND = ['COAL', 'OILGAS', 'ORES', 'COKE', 'METAL', 'FABMET', 'ELEC',
       'WASTE', 'CONSTR', 'TRADE', 'TRANSP', 'WAREH', 'OTHERS']
IND12 = IND[:-1]
FD = ['HH', 'NPISH', 'GOV', 'GFCF', 'INVNT', 'DPABR']
COLS = [*IND, 'TOTAL_IC', *FD, 'OUTPUT', 'EXPORT', 'IMPORT']
NAMES = {'COAL': 'Coal mining', 'OILGAS': 'Oil and gas extraction', 'ORES': 'Metal ores mining',
         'COKE': 'Coke and refined petroleum products', 'METAL': 'Basic iron and steel (C24A)',
         'FABMET': 'Fabricated metal products', 'ELEC': 'Electricity, gas and steam',
         'WASTE': 'Water supply and waste management', 'CONSTR': 'Construction',
         'TRADE': 'Wholesale and retail trade', 'TRANSP': 'Transport (H49+H50+H51)',
         'WAREH': 'Warehousing and support activities (H52)', 'OTHERS': 'Other industries and services',
         'TOTAL_IC': 'Total domestic intermediate inputs (column sums)',
         'IMP_IC': 'Imported intermediate inputs', 'TLS': 'Taxes less subsidies on products',
         'VA': 'Gross value added', 'OUTPUT': 'Output (column)'}
log = []   # журнал виправлень: (country, year, what, detail)

def rel(a, e):
    a, e = np.asarray(a, float), np.asarray(e, float)
    m = ~np.isnan(a) & ~np.isnan(e)
    return float(np.max(np.abs(a[m] - e[m]) / np.maximum(np.abs(e[m]), 1))) if m.any() else np.nan


# %% [markdown]
# ## 1. Читання та агрегування
#
# Розбір за кодом ICIO в назві (`UKR_C24A` → країна `UKR`, код `C24A`). Колонки, що належать іншій країні (`IRL_HFCE`, `VNM_HFCE`) або кодам без відповідника (`A01`–`A03` у SAU 2015), не переносяться й записуються в журнал.


# %%
def split_label(lbl, country):
    head, _, tail = str(lbl).partition('_')
    if not tail:                         # 'OUT'
        return country, head.upper()
    return head.upper(), tail.upper()

def read_block(path):
    country, year = path.stem.split('-')
    year = int(year)
    df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig', index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    cmap, rmap = {}, {}
    for c in df.columns:
        cc, code_ = split_label(c, country)
        if cc != country or code_ not in ICIO_MAP:
            log.append((country, year, 'dropped column', f'{c}: sum = {df[c].sum():.2f}'))
        elif ICIO_MAP[code_]:
            cmap[c] = ICIO_MAP[code_]
    for r in df.index:
        cc, code_ = split_label(r, country)
        if ICIO_MAP.get(code_):
            rmap[r] = ICIO_MAP[code_]
    d = df.loc[list(rmap), list(cmap)]
    d = d.T.groupby(cmap).sum(min_count=1).T.groupby(rmap).sum(min_count=1)
    return country, year, d.reindex(index=[*IND, 'TOTAL_IC'], columns=COLS)

blocks = {}
for p in sorted(RAW.glob('*-*.csv')):
    c, y, d = read_block(p)
    blocks[(c, y)] = d
print(len(blocks), 'блоків:', sorted(blocks))


# %% [markdown]
# ## 2. Виправлення
#
# | Блок | Проблема у джерелі | Виправлення | Статус |
# |---|---|---|---|
# | UKR 2021 | рядок TOTAL_IC менший за суму стовпців рівно на рядок ORES (рядок руд пропущено при підсумовуванні) | TOTAL_IC за стовпцями перераховано як суму 13 рядків | `ok` |
# | IND 2022 | рядок і стовпець OTHERS хибні (OTHERS→METAL 141 714 при 10 836 у 2021; OTHERS×OTHERS < 0) | OTHERS відновлено як залишок: TOTAL_IC − Σ 12 галузей; OTHERS×OTHERS невідомий | `reconstructed` |
# | USA 2022 | рядок OTHERS хибний (OTHERS→COKE 439 237 при 36 371 у 2021); стовпець узгоджений | рядок OTHERS відновлено як залишок; OTHERS×OTHERS невідомий | `reconstructed` |
# | SAU 2022 | рядок і стовпець OTHERS хибні | відновлено як залишок | `unreliable` |
#
# Блоки 2022 року потім **замінюються** даними повної таблиці ICIO (розділ 3); звірка показала, що відновлені залишком значення OTHERS збігаються з ICIO, а помилка вивантаження полягала також у заниженому експорті.
# | SAU 2015 | колонки A01–A03 без рядків, не входять у TOTAL_IC | не переносяться (узгоджено з TOTAL_IC) | `ok` (примітка) |
# | IND 2021, USA 2021 | колонки `IRL_HFCE`, `VNM_HFCE` — кінцевий попит інших країн | не переносяться | `ok` (примітка) |
#
# Залишок рахується лише для перетинів OTHERS з 12 галузями; OTHERS×OTHERS і підсумки для OTHERS лишаються порожніми, бо для них немає незалежного контрольного значення.


# %%
STATUS = {k: 'ok' for k in blocks}
NOTE = {k: [] for k in blocks}
ROW_NOTE = {}

def fix_total_row(c, y):
    d = blocks[(c, y)]
    old = d.loc['TOTAL_IC', IND].copy()
    new = d.loc[IND, IND].sum()
    gap = new - old
    d.loc['TOTAL_IC', IND] = new
    log.append((c, y, 'TOTAL_IC row recomputed', f'max change {gap.abs().max():.2f}; '
                f'gap equals ORES row: {np.allclose(gap, d.loc["ORES", IND], atol=0.01)}'))
    NOTE[(c, y)].append('TOTAL_IC row recomputed as column sums (ORES row omitted in source)')

def reconstruct_others(c, y, row=True, col=True, status='reconstructed'):
    d = blocks[(c, y)]
    if row:
        old = d.loc['OTHERS', IND12].copy()
        d.loc['OTHERS', IND12] = d.loc['TOTAL_IC', IND12] - d.loc[IND12, IND12].sum()
        log.append((c, y, 'OTHERS row reconstructed',
                    f'OTHERS→METAL {old["METAL"]:.0f} → {d.loc["OTHERS", "METAL"]:.0f}'))
        ROW_NOTE[(c, y, 'OTHERS')] = 'row reconstructed as residual (TOTAL_IC − Σ 12 industries)'
    if col:
        old = d.loc[IND12, 'OTHERS'].copy()
        d.loc[IND12, 'OTHERS'] = d.loc[IND12, 'TOTAL_IC'] - d.loc[IND12, IND12].sum(axis=1)
        log.append((c, y, 'OTHERS column reconstructed',
                    f'METAL→OTHERS {old["METAL"]:.0f} → {d.loc["METAL", "OTHERS"]:.0f}'))
    d.loc['OTHERS', 'OTHERS'] = np.nan
    d.loc['OTHERS', 'TOTAL_IC'] = np.nan
    d.loc['TOTAL_IC', ['OTHERS', 'TOTAL_IC']] = np.nan
    neg = [(i, j) for i in IND for j in IND if d.loc[i, j] < 0]
    if neg:
        log.append((c, y, 'negative residuals', str(neg)))
    STATUS[(c, y)] = status
    NOTE[(c, y)].append(f'OTHERS {"row" if row else ""}{" and " if row and col else ""}{"column" if col else ""} '
                        'reconstructed as residual; OTHERS×OTHERS unknown')

fix_total_row('UKR', 2021)
reconstruct_others('IND', 2022, row=True, col=True)
reconstruct_others('USA', 2022, row=True, col=False)
reconstruct_others('SAU', 2022, row=True, col=True, status='unreliable')
NOTE[('SAU', 2022)].append('residual values implausible vs 2021; re-extract from ICIO')

for (c, y, what, detail) in log:
    if what == 'dropped column':
        NOTE[(c, y)].append(f'dropped source column {detail.split(":")[0]}')

print(pd.DataFrame(log, columns=['country', 'year', 'action', 'detail']))


# %% [markdown]
# ## 3. Блоки з повної таблиці ICIO
#
# Для років, за які є повна таблиця, блоки чотирьох країн будуються безпосередньо з неї й **замінюють** вивантаження. Визначення:
#
# | Показник | Обчислення з ICIO |
# |---|---|
# | Z (13×13) | вітчизняні рядки країни × вітчизняні стовпці галузей, агреговані за відповідністю кодів (усі коди поза переліком → OTHERS, зокрема A01–A03, C24B, H53, I…T) |
# | кінцевий попит | вітчизняні рядки × власні стовпці HFCE…DPABR |
# | `EXPORT` | вітчизняні рядки × усі стовпці інших країн (проміжні та кінцеві) |
# | `IMPORT` | іноземні рядки продукту, використані в країні (проміжне + кінцеве), довідково |
# | рядок `IMP_IC` | імпортні проміжні витрати галузі-споживача (іноземні рядки × її стовпець) |
# | рядок `TLS` | податки мінус субсидії на продукти, сплачені галуззю-споживачем |
# | рядок `VA` | валова додана вартість (рядок VA ICIO) |
# | рядок `OUTPUT` | випуск за стовпцем (рядок OUT ICIO) |
#
# Тотожності: `TOTAL_IC + FD + EXPORT = OUTPUT` (за рядком) і `TOTAL_IC + IMP_IC + TLS + VA = OUTPUT` (за стовпцем).


# %%
FULL = Path('raw/icio_full')
COUNTRIES = ['IND', 'SAU', 'UKR', 'USA']
MAP_FULL = {k: v for k, v in ICIO_MAP.items() if v in IND}          # B05…H52; решта → OTHERS
FDMAP = {'HFCE': 'HH', 'NPISH': 'NPISH', 'GGFC': 'GOV', 'GFCF': 'GFCF', 'INVNT': 'INVNT', 'DPABR': 'DPABR'}
EXTRA_ROWS = ['IMP_IC', 'TLS', 'VA', 'OUTPUT']

def sector(code_):
    return MAP_FULL.get(code_, 'OTHERS')

def aggregate_full(df, cc):
    ind_rows = [r for r in df.index if '_' in r]
    ind_set = set(ind_rows)
    dom_rows = [r for r in ind_rows if r.startswith(cc + '_')]
    for_rows = [r for r in ind_rows if not r.startswith(cc + '_')]
    ind_cols = [c for c in df.columns if c in ind_set]
    dom_cols = [c for c in ind_cols if c.startswith(cc + '_')]
    fd_cols = [c for c in df.columns if c not in ind_set and c != 'OUT']
    dom_fd = [c for c in fd_cols if c.startswith(cc + '_')]
    for_use = [c for c in ind_cols + fd_cols if not c.startswith(cc + '_')]

    rmap = {r: sector(r.split('_', 1)[1]) for r in dom_rows}
    cmap = {c: sector(c.split('_', 1)[1]) for c in dom_cols}
    fdmap = {c: FDMAP[c.split('_', 1)[1]] for c in dom_fd}
    pmap = {r: sector(r.split('_', 1)[1]) for r in for_rows}
    by_rows = lambda d, m: d.groupby(m).sum()
    by_cols = lambda d, m: d.T.groupby(m).sum().T

    Z = by_cols(by_rows(df.loc[dom_rows, dom_cols], rmap), cmap).reindex(index=IND, columns=IND)
    blk = pd.DataFrame(index=[*IND, 'TOTAL_IC', *EXTRA_ROWS], columns=COLS, dtype=float)
    blk.loc[IND, IND] = Z.values
    blk.loc[IND, 'TOTAL_IC'] = Z.sum(axis=1).values
    blk.loc['TOTAL_IC', IND] = Z.sum().values
    blk.loc['TOTAL_IC', 'TOTAL_IC'] = Z.values.sum()
    blk.loc[IND, FD] = by_cols(by_rows(df.loc[dom_rows, dom_fd], rmap), fdmap).reindex(index=IND, columns=FD).values
    blk.loc[IND, 'OUTPUT'] = by_rows(df.loc[dom_rows, 'OUT'], rmap).reindex(IND).values
    blk.loc[IND, 'EXPORT'] = by_rows(df.loc[dom_rows, for_use].sum(axis=1), rmap).reindex(IND).values
    blk.loc[IND, 'IMPORT'] = by_rows(df.loc[for_rows, dom_cols + dom_fd].sum(axis=1), pmap).reindex(IND).values
    blk.loc['IMP_IC', IND] = by_cols(df.loc[for_rows, dom_cols].sum().to_frame().T, cmap).iloc[0].reindex(IND).values
    for r_src, r_dst in [('TLS', 'TLS'), ('VA', 'VA'), ('OUT', 'OUTPUT')]:
        blk.loc[r_dst, IND] = by_cols(df.loc[[r_src], dom_cols], cmap).iloc[0].reindex(IND).values
    return blk

# блоки з вивантажень доповнюються порожніми рядками IMP_IC, TLS, VA і рядком OUTPUT (= колонка OUTPUT)
for k, d in blocks.items():
    d = d.reindex(index=[*IND, 'TOTAL_IC', *EXTRA_ROWS])
    d.loc['OUTPUT', IND] = d.loc[IND, 'OUTPUT'].values
    blocks[k] = d

for path in sorted(FULL.glob('*_SML.csv')):
    year = int(path.stem.split('_')[0])
    full = pd.read_csv(path, index_col=0)
    for cc in COUNTRIES:
        new = aggregate_full(full, cc)
        if (cc, year) in blocks:     # звірка з попереднім вивантаженням
            old = blocks[(cc, year)]
            z_dev = rel(old.loc[IND, IND].values.ravel(), new.loc[IND, IND].values.ravel())
            e_dev = rel(old.loc[IND12, 'EXPORT'], new.loc[IND12, 'EXPORT'])
            log.append((cc, year, 'replaced by full ICIO', f'Z max rel. diff vs extract {z_dev:.2e}; '
                        f'EXPORT max rel. diff {e_dev:.1%} (extract understated exports)'))
        blocks[(cc, year)] = new
        STATUS[(cc, year)] = 'ok'
        NOTE[(cc, year)] = [f'built from full OECD ICIO {year} ({path.name})']
        ROW_NOTE = {k: v for k, v in ROW_NOTE.items() if k[:2] != (cc, year)}
    del full

print(pd.DataFrame(log, columns=['country', 'year', 'action', 'detail']).tail(4))


# %% [markdown]
# ## 4. Перевірки після виправлень
#
# * `colsum_dev` — Σ рядків проти рядка TOTAL_IC (за стовпцями з відомим підсумком);
# * `rowsum_dev` — Σ стовпців проти колонки TOTAL_IC (12 галузей);
# * `balance_dev` — `TOTAL_IC + кінцевий попит + експорт` проти `OUTPUT`; розбіжність є і в джерелі (найбільша — H51, авіатранспорт), її не виправлено;
# * `metal_coef_sum` — сума коефіцієнтів прямих витрат металургії (має бути < 1);
# * `col_balance_dev` — `TOTAL_IC + IMP_IC + TLS + VA` проти `OUTPUT` (лише для блоків з повної ICIO);
# * `metal_va_share` — частка ВДВ у випуску металургії.


# %%
rows = []
for (c, y), d in sorted(blocks.items()):
    x = d.loc[IND12, 'OUTPUT']
    bal = d.loc[IND12, 'TOTAL_IC'] + d.loc[IND12, FD].sum(axis=1) + d.loc[IND12, 'EXPORT']
    ok = x > 0
    rows.append(dict(
        country=c, year=y, status=STATUS[(c, y)],
        colsum_dev=rel(d.loc[IND, IND].sum(min_count=13), d.loc['TOTAL_IC', IND]),
        rowsum_dev=rel(d.loc[IND12, IND].sum(axis=1), d.loc[IND12, 'TOTAL_IC']),
        balance_dev=rel(bal[ok], x[ok]),
        balance_worst=((bal - x).abs() / x.where(ok)).idxmax(),
        metal_coef_sum=d.loc[IND, 'METAL'].sum() / d.loc['METAL', 'OUTPUT'],
        negatives=int((d.loc[IND, IND] < 0).sum().sum()),
        col_balance_dev=rel(d.loc[['TOTAL_IC', 'IMP_IC', 'TLS', 'VA'], IND].sum(min_count=4), d.loc['OUTPUT', IND]),
        metal_va_share=d.loc['VA', 'METAL'] / d.loc['OUTPUT', 'METAL'],
    ))
report = pd.DataFrame(rows)
report.to_csv(OUT / 'validation_report.csv', index=False)
print(report.round(4))


# %% [markdown]
# ## 5. Запис бази


# %%
FROM_FULL = {(c, y) for (c, y, a, _) in log if a == 'replaced by full ICIO'}
for k in blocks:
    if k not in FROM_FULL:
        NOTE[k].append(f'source: country extract (no VA/TLS/IMP_IC); EXPORT understated, '
                       f'row balance gap up to {report.set_index(["country", "year"]).loc[k, "balance_dev"]:.1%}')

recs = []
for (c, y), d in sorted(blocks.items()):
    b = d.copy()
    b.insert(0, 'row', b.index)
    b.insert(0, 'year', y)
    b.insert(0, 'country', c)
    b.insert(3, 'row_name', b.row.map(NAMES))
    b.insert(4, 'row_type', np.select([b.row.isin(IND), b.row == 'TOTAL_IC'], ['industry', 'total'], 'input'))
    b['row_note'] = [ROW_NOTE.get((c, y, r), '') for r in b.row]
    b['block_status'] = STATUS[(c, y)]
    b['block_note'] = '; '.join(NOTE[(c, y)])
    recs.append(b)

base = pd.concat(recs, ignore_index=True)
base.to_csv(OUT / 'mbc_base.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
pd.DataFrame(log, columns=['country', 'year', 'action', 'detail']).to_csv(OUT / 'corrections_log.csv', index=False)

print(base.shape)
print(base.pivot_table(index='country', columns='year', values='block_status', aggfunc='first'))


# %% [markdown]
# ## 6. Контроль: вхідний слід і частка ВДВ металургії


# %%
u = base.query('country == "UKR"').set_index(['year', 'row'])
coef = pd.DataFrame({y: u.loc[y].loc[IND, 'METAL'] / u.loc[(y, 'METAL'), 'OUTPUT'] for y in [2015, 2019, 2021, 2022]})
print(coef.loc[['COAL', 'OILGAS', 'ORES', 'METAL', 'WASTE']].round(5))

va = (base.query('row == "VA"').set_index(['country', 'year'])['METAL'] /
      base.query('row == "OUTPUT"').set_index(['country', 'year'])['METAL'])
print('\nЧастка ВДВ у випуску металургії (лише блоки з повної ICIO):')
print(va.unstack().round(3))