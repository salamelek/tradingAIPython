"""
Given a time interval, it creates a filler csv file with missing data
it uses the binance API to get the missing data
"""


import requests


symbol = "XRPUSDT"
interval = "5m"
startTime = 1606780500000
endTime = 1606781100000

limit = int((endTime - startTime - 300000) / 300000)


url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={startTime + 300000}&limit={limit}"

# make the api request
response = requests.get(url)

# get the data
data = response.json()


row = data[0]
print(f"{row[0]},{float(row[1])},{float(row[2])},{float(row[3])},{float(row[4])},{float(row[5])}")
exit()


# write the data to a csv file
fileName = f"{symbol}-{interval}-2020-01.filler.csv"

with open(fileName, "w") as file:
    # write header
    file.write("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n")

    for row in data:
        file.write(f"{row[0]},{float(row[1])},{float(row[2])},{float(row[3])},{float(row[4])},{float(row[5])}\n")