import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from model import Kronos, KronosTokenizer, KronosPredictor
import matplotlib.dates as mdates

def plot_history_and_prediction(kline_df, pred_df, y_timestamp):
    # 将所有时间戳转换为 naive 时间戳，去掉时区信息
    kline_df['timestamp'] = kline_df['timestamp'].dt.tz_localize(None)
    pred_df.index = pred_df.index.tz_localize(None)
    y_timestamp = y_timestamp.dt.tz_localize(None)

    # 创建一个图表，分为左右两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # 绘制左边的历史数据（历史 Close Price）
    ax1.plot(kline_df['timestamp'], kline_df['close'], label='Close Price', color='blue', linewidth=1.5)
    ax1.set_title('Historical Data', fontsize=16)
    ax1.set_xlabel('Timestamp', fontsize=14)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.grid(True)
    ax1.legend(loc='upper left', fontsize=12)

    # 绘制右边的预测数据（预测 Close Price）
    ax2.plot(y_timestamp, pred_df['close'], label='Predicted Close Price', color='red', linewidth=1.5)
    ax2.set_title('Predicted Data', fontsize=16)
    ax2.set_xlabel('Timestamp', fontsize=14)
    ax2.set_ylabel('Close Price', fontsize=14)
    ax2.grid(True)
    ax2.legend(loc='upper left', fontsize=12)

    # 格式化 X 轴为日期时间
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def calculate_percentage_change(actual, predicted):
    return (predicted - actual) / actual * 100

# 获取 ETH/USDT 的历史数据
# exchange = ccxt.binance()  # 初始化 Binance API
exchange = ccxt.binance({
    'urls': {
        'api': {
            'public': 'https://zysipeyigmoy.eu-central-1.clawcloudrun.com/api/v3',
            'private': 'https://zysipeyigmoy.eu-central-1.clawcloudrun.com/api/v3',
        },
        'fapiPublic': 'https://zysipeyigmoy.eu-central-1.clawcloudrun.com/fapi/v1',
        'fapiPrivate': 'https://zysipeyigmoy.eu-central-1.clawcloudrun.com/fapi/v1',
        'dapiPublic': 'https://zysipeyigmoy.eu-central-1.clawcloudrun.com/dapi/v1',
        'dapiPrivate': 'https://zysipeyigmoy.eu-central-1.clawcloudrun.com/dapi/v1',
        'sapiPublic': 'https://zysipeyigmoy.eu-central-1.clawcloudrun.com/sapi/v1',
        'sapiPrivate': 'https://zysipeyigmoy.eu-central-1.clawcloudrun.com/sapi/v1',
    },
    'timeout': 30000,  # 30秒超时
    'enableRateLimit': True,  # 启用速率限制
    'hostname': 'zysipeyigmoy.eu-central-1.clawcloudrun.com',
    'options': {
        'defaultType': 'spot',  # 明确使用现货交易
    }
})
symbol = 'ETH/USDT'  # 交易对
timeframe = '15m'  # 时间框架：15分钟
limit = 400  # 获取的数据条数，最多获取1000个数据点

# 获取数据
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

# 转换为 Pandas DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# 将时间戳从 UTC 转换为中国标准时间 (CST)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # 将时间戳转换为 datetime 格式
df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')  # 转换为 CST 时区

# 只保留最新的 400 条数据
df = df.tail(limit)

# 显示前几行数据，确保时间戳已经转换
print("历史数据的时间戳：")
print(df)

# 加载模型和 Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 初始化预测器
# predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

# 准备数据
lookback = limit  # 使用最新的400条数据
pred_len = 100  # 预测未来2小时（8个15分钟数据点）

x_df = df[['open', 'high', 'low', 'close', 'volume']]
x_timestamp = pd.Series(df['timestamp'])  # 获取最新的时间戳，确保顺序正确

# 生成未来预测的时间戳（接在最后一根K线之后）
last_time = df['timestamp'].iloc[-1]
y_timestamp = pd.Series(pd.date_range(
    start=last_time + pd.Timedelta(minutes=15),
    periods=pred_len,
    freq='15min',
    tz="Asia/Shanghai"  # 明确设置时区为 Asia/Shanghai
))

# 进行预测
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 确保预测数据的时间戳正确设置
pred_df.index = y_timestamp  # 设置预测数据的时间戳为 y_timestamp

# 打印预测数据
print("预测数据：")
print(pred_df)

# 计算涨跌幅度（与历史数据最后一个 close 比较）
last_close = df['close'].iloc[-1]  # 历史数据最后一个 `close`

# 计算每个预测时间点的涨跌幅度
pred_changes = pred_df['close'].apply(lambda x: calculate_percentage_change(last_close, x))

print("\n预测的涨跌幅度：")
print(pred_changes)

# 可视化预测数据与历史数据
plot_history_and_prediction(df, pred_df, y_timestamp)

# 输出详细的预测和涨跌幅度数据
print("\n详细预测结果及涨跌幅度：")
for i in range(pred_len):
    print(f"预测时间: {y_timestamp.iloc[i]} - 预测的 Close Price: {pred_df['close'].iloc[i]:.2f} - 涨跌幅度: {pred_changes.iloc[i]:.2f}%")
