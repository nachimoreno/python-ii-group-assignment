from ..PySimFin import PySimFin

def PySimFin_test():
    test = PySimFin()
    df = test.get_share_prices('AMZN')
    print(df)
    print('\n')
    df = test.get_financial_statement('AMZN', statement='pl,bs,cf,derived', start='2025-01-01', end='2026-03-22')
    print(df)

if __name__ == "__main__":
    PySimFin_test()