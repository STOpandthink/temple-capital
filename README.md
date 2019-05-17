# temple-capital
Public repository for Temple Capital code

If you have the password for the hourly bars:  
`openssl enc -d -aes-256-cbc -in hours_btc_usd_bitmex.enc -out hours_btc_usd_bitmex.csv`  
Then in the notebook:  
`df = notebook_utils.load_time_bars_df(coin, exchange_name="bitmex", bar_size="hours")`  
