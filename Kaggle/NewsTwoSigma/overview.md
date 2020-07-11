# Description



August 2019 Update: this competition is closed and is no longer accepting submissions. Thanks for participating!**

> 2019年8月更新：本比赛已关闭，不再接受参赛作品。感谢您的参与！

Can we use the content of news analytics to predict stock price performance? The ubiquity of data today enables investors at any scale to make better investment decisions. The challenge is ingesting and interpreting the data to determine which data is useful, finding the signal in this sea of information. Two Sigma is passionate about this challenge and is excited to share it with the Kaggle community

> 我们可以使用新闻分析的内容来预测股价表现吗？如今，无处不在的数据使各种规模的投资者都能做出更好的投资决策。挑战是摄取和解释数据以确定哪些数据有用，从而在这片信息海中寻找信号。两位Sigma对此挑战充满热情，并很高兴与Kaggle社区分享这一挑战。

As a scientifically driven investment manager, Two Sigma has been applying technology and data science to financial forecasts for over 17 years. Their pioneering advances in big data, AI, and machine learning have pushed the investment industry forward. Now, they're eager to engage with Kagglers in this continuing pursuit of innovation.

> 作为一家科学驱动的投资经理，Two Sigma一直将技术和数据科学应用于财务预测已有17年以上。他们在大数据，人工智能和机器学习方面的开拓性进步推动了投资行业的发展。现在，他们渴望与Kagglers合作，不断追求创新。

By analyzing news data to predict stock prices, Kagglers have a unique opportunity to advance the state of research in understanding the predictive power of the news. This power, if harnessed, could help predict financial outcomes and generate significant economic impact all over the world.

> 通过分析新闻数据以预测股票价格，Kagglers有一个独特的机会来推动研究状态以了解新闻的预测能力。如果加以利用，这种力量将有助于预测财务结果并在全世界范围内产生重大的经济影响。

Data for this competition comes from the following sources:

- Market data provided by Intrinio.
- News data provided by Thomson Reuters. Copyright Thomson Reuters, 2017. All Rights Reserved. Use, duplication, or sale of this service, or data contained herein, except as described in the Competition Rules, is strictly prohibited.

> 该比赛的数据来自以下来源： 
>
> - 市场数据由Intrinio提供。 
> - 汤森路透提供的新闻数据。汤森路透，2017年版权所有。保留所有权利。严格禁止使用，复制或出售本服务或此处包含的数据，除非《竞赛规则》另有规定。

# Evaluation

In this competition, you must predict a signed confidence value,$\hat{y}_{ti} \in [-1,1]$ , which is multiplied by the market-adjusted return of a given `assetCode` over a ten day window. If you expect a stock to have a large positive return--compared to the broad market--over the next ten days, you might assign it a large, positive `confidenceValue` (near 1.0). If you expect a stock to have a negative return, you might assign it a large, negative `confidenceValue` (near -1.0). If unsure, you might assign it a value near zero。

> 在这个竞赛中，你必须预测一个签名的信心值，$\hat{y}_{ti} \in [-1,1]$，它乘以给定的“资产代码”在10天内的经市场调整的回报率。如果你预计一只股票在未来10天内会比大盘有较大的正回报，你可能会给它一个较大的正“信心值”(接近1.0)。如果你预计一只股票会有负回报，你可能会给它一个很大的负“信心值”(接近-1.0)。如果不确定，可以将其赋值为接近零的值。

For each day in the evaluation time period, we calculate:
$$
x_t = \sum_i \hat{y}_{ti}  r_{ti}  u_{ti},
$$
where $r_{ti}$ is the 10-day market-adjusted leading return for day t for instrument i, and $u_{ti}$ is a 0/1 `universe` variable (see the data description for details) that controls whether a particular asset is included in scoring on a particular day.

其中$r_{ti}$是 instrument i 第 t 天的10天经市场调整的领先收益，$u_{ti}$是一个0/1的“universe”变量(详见数据描述)，控制某一特定资产是否包括在某一特定日期的得分中。

Your submission score is then calculated as the mean divided by the standard deviation of your daily $x_t$ values:

