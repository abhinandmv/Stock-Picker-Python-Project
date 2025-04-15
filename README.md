# 📈 Indian Stock Market Analysis Dashboard

A **Dash** application for analyzing Indian stocks (NSE) using historical data, intraday charts, and simple future price predictions. Built with **Plotly Dash**, **Bootstrap**, and **yfinance**.

---

## 🚀 Features

- 🔍 Enter NSE stock symbols (e.g., `RELIANCE.NS`)
- 📊 View historical opening and closing price graphs
- 📉 Live intraday stock prices (5-min interval)
- 📋 Daily stock data summary
- 🧾 Display of financial metrics (via `yfinance`)
- 🔮 Dummy future price predictions (next 5 days)
- 🌙 Clean dark-themed UI using Bootstrap

---

## 📁 Project Structure<br>
Stock-Market-Dashboard/ <br>
├── project.py # Main Dash application <br>
└── README.md # Project documentation<br>


---

## ⚙️ Installation & Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/stock-market-dashboard.git
   cd stock-market-dashboard
   pip install dash plotly dash-bootstrap-components yfinance numpy
   python project.py
   ```
## ⚠️ Known Issues
```
The app currently uses the yfinance package to fetch live and historical stock data. There are occasional issues where:
```
- Financial data (like .financials) may return empty
- Intraday data may not load if the market is closed or API limits are hit
- Certain ticker symbols may not return expected results
- These issues are external and depend on Yahoo Finance API limitations.

## 💡 Future Enhancements
- Add real-time price streaming
- Cache stock data to improve performance
- Add technical indicators (e.g., RSI, MACD)
- Allow CSV download of stock history

## 📄 License
- This project is open-source and available under the MIT License.




