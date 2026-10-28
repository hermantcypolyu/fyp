# 廣東話講稿：跟住 Code Walkthrough 介紹 Pipeline

> 呢份講稿按 [`code_walkthrough.md`](./code_walkthrough.md) 嘅次序編排。你可以一邊讀每節，一邊展示同一節提到嘅程式碼。對象係第一次認識本 project 嘅聽眾；建議時間約 8 至 10 分鐘。

## 開場：今日想解答嘅問題

大家好，今日我會介紹我個 FYP 入面嘅 coding-agent pipeline。

我想研究嘅係：可唔可以俾一個 AI agent 接手一條 Python 程式修正題，自己睇 code、用工具改 code、跑測試，再根據測試結果計分，最後將呢個分數用喺模型訓練？

所以如果舉例既話，
我會用「考生做題」做比喻：
題目係考卷，
Agent 係考生，
工具係考生可以用嘅工具，
Verifier 係改卷員，
而 Reward 就係分數。

## 1. 題目同資料集

首先睇題目。現時 project 入面有三十六條 task：二十六條 training tasks，同十條 held-out tasks。

Held-out 即係保留起嚟嘅測試題，唔應該用佢嘅答案去訓練；之後可以用嚟獨立檢查系統表現。

每條 task 都有題目描述、一份 Agent 可以修改嘅程式、一啲可見 smoke tests，同一套私下保管嘅評分 tests。題目來源唔係全部一樣：有為 project 編寫嘅、有按常見程式設計模式整理嘅，亦有由真實開源 project 整理嘅。

所以準確啲講，呢個係 FYP 嘅內部 benchmark，採用 SWE-style 嘅可執行題目設計；但佢唔係直接下載返嚟嘅 SWE-Gym Lite dataset，亦唔係 SWE-Gym Lite 原本嘅資料格式。

## 2. Agent Harness

第二部分係 Agent Harness。你可以當佢係一個流程協調員：佢負責將題目交畀模型，將模型提出嘅工具要求交去執行，再將工具結果交返畀模型。

Agent 唔係一次過寫晒答案，而係可以做幾輪：睇資料、作決定、用工具，之後再根據結果決定下一步。呢個 project 嘅 wrapper 亦會限制每條題目最多可以做幾多輪。

## 3. 工具

第三部分係工具。模型提出「讀檔」、「改檔」或者「跑測試」等要求；工具箱會檢查要求，再執行合適嘅操作，然後將結果回覆畀 Agent。

呢個 project 有七種工具：列出檔案、讀檔、寫檔、編輯檔案、替換指定程式行、跑測試，同提交答案。

要留意，Agent 跑到嘅係可見 smoke tests。提交答案只係代表做完，唔會即時將 hidden tests 題目內容交畀 Agent。

## 4. 一次作答同 Sandbox

第四部分係一次作答過程，同埋 Agent 做題嘅工作區。

每次做題都有自己嘅工作副本，叫 sandbox。可以想像成每位考生有自己一張工作枱；Agent 改嘅係呢份副本，唔係直接改動原本 benchmark。

目前內部 benchmark 用 filesystem sandbox。工作區會有題目程式同可見 smoke tests，但唔會放入 private grading tests。提交之後，系統先另外準備評分用嘅副本。

Docker 同 sandbox 都係講程式喺邊度執行，唔係講題目由邊度嚟。Project 有一條獨立嘅 Docker-backed SWE-Gym Lite pilot path；但呢條 path 嘅存在，唔代表目前內部 benchmark 係用 Docker，亦唔代表目前已經用緊 SWE-Gym Lite。

簡單講：benchmark 講「做邊啲題」；sandbox 或 Docker 講「喺邊度做題」。

## 5. Verifier 同 Reward

第五部分係改卷。Agent 提交之後，Verifier 會準備乾淨嘅評分副本，喺入面執行 private tests，同時檢查預先定義嘅規則。Agent 收到評分結果，但唔會因此見到每條 hidden test 嘅內容。

Reward 會衡量 Agent 比原本未修改嘅程式改善咗幾多。簡單公式係：

```text
Reward = (Agent 通過測試數 - 原本通過測試數)
         ÷ (總測試數 - 原本通過測試數)
最後會限制喺 -1 至 1 之間
```

例如總共有十條測試，原本程式通過四條，Agent 改完之後通過七條。佢改善空間係六條，今次多通過三條，所以 Reward 係三除六，即係 0.5。呢個係方便理解嘅例子，唔係某次實際實驗結果。

如果觸發預先定義嘅 hard gate，Reward 就會係負一。呢個分數將測試結果轉成一個可比較嘅數字，之後可以作為訓練訊號。

## 6. Rollout Adapter

第六部分係將做題過程整理成訓練資料。

Agent 做一題嘅互動過程叫 rollout。過程入面有模型自己產生嘅內容，亦有工具回覆；工具回覆唔係模型寫出嚟嘅答案，所以唔應該當成模型行為去訓練。

Rollout adapter 會挑出有模型生成 token 記錄嘅部分，並檢查需要嘅 token 同舊模型機率資料齊唔齊。用考試比喻，佢會分清楚考生寫嘅答案，同監考員提供嘅資料。

## 7. Group-relative Advantage

第七部分係比較同一組嘗試嘅表現，計算 Advantage。可以將 Advantage 理解成：「呢次結果比同組平均高幾多，或者低幾多。」

公式係：

$$
A_i = \frac{R_i-\mu_R}{\sigma_R+10^{-6}},
\qquad
\mu_R=\frac{1}{G}\sum_{j=1}^{G}R_j,
\qquad
\sigma_R=\sqrt{\frac{1}{G}\sum_{j=1}^{G}(R_j-\mu_R)^2}
$$

呢度 $R_i$ 係第 $i$ 次嘗試嘅 Reward，$G$ 係同一組嘗試嘅數量；$\mu_R$ 係平均 Reward，$\sigma_R$ 係呢組 Reward 嘅母體標準差。分母加上 $10^{-6}$，係一個非常細嘅數值，令計算更穩定。

高過平均嘅嘗試會有正數，低過平均就會有負數。如果一組嘗試嘅分數全部完全一樣，系統就分唔到邊種做法相對好啲，所以呢個 optimizer smoke test 會停止，而唔會假裝有比較訊號。

## 8. Policy Loss 同模型更新

第八部分係由比較結果計算更新方向，再更新 LoRA 參數。

程式會比較新模型同產生 rollout 時嘅舊模型，對同一個 token 有幾大傾向。呢個比較叫 ratio。之後會用 clipping 限制 ratio 對更新嘅影響，避免單次更新過大。

如果 ratio 等於一，新舊模型對該 token 嘅傾向相同；大過一代表新模型相對更傾向，細過一就代表相對冇咁傾向。

實際 loss 會將呢個 ratio、Advantage 同 clipping 結合，再對所有可訓練 token 取平均。用數學式寫出嚟係：

$$
\mathcal{L}(\theta)
= -\frac{1}{N_{\text{train}}}
\sum_{i=1}^{G}\sum_{t\in\mathcal{T}_i}
\min\left(
r_{i,t}(\theta)A_i,\;
\operatorname{clip}\left(r_{i,t}(\theta),1-\epsilon,1+\epsilon\right)A_i
\right)
$$

呢度 $i$ 代表第 $i$ 次嘗試，$t$ 代表嗰次嘗試入面一個可訓練嘅生成 token；$A_i$ 係嗰次嘗試嘅 Advantage；$r_{i,t}(\theta)$ 比較新舊模型對該 token 嘅機率；而 $N_{\text{train}}$ 係所有可訓練 token 嘅總數。`clip` 會將 ratio 限制喺 $1-\epsilon$ 同 $1+\epsilon$ 之間。

程式亦會喺計算指數之前，將 log-probability 差值限制喺 $-20$ 至 $20$，避免數值溢位；呢個保護同公式入面限制 ratio 嘅 clipping 係兩件唔同嘅事。

簡單理解：表現較好嘅生成內容會提供正向訊號，表現較差嘅會提供負向訊號；clipping 就限制每次訊號可以推動模型幾多。之後 `backward()` 計算 gradient，`optimizer.step()` 先真正更新 LoRA 參數。

要講清楚範圍：呢次 optimizer smoke test 實作咗上面所講嘅 clipped objective，但冇實作 KL penalty，亦冇 value model 或 critic。因此介紹呢次實作時，唔好將呢兩樣講成已經做咗。

## 9. 儲存、重新載入同目前證據

第九部分係檢查更新有冇真正發生，同埋 checkpoint 有冇保存成功。

Phase F4 用一條內部 training task 做咗一次 optimizer update：有八條 rollouts、四十八個可訓練 turns，同一千三百九十個可訓練 tokens。結果有三百九十二個 LoRA parameter tensors 改變；儲存再重新載入之後，參數最大差異係 0.0。

呢個結果證明單步更新同 checkpoint 保存、重新載入嘅流程成功運作；但唔代表模型已經更識做 coding。要證明模型能力有進步，仲要用公平嘅更新前後比較，再用 held-out tasks 作獨立評估。

## 10. 未來題目來源：SWE-Gym Lite

最後講未來方向。SWE-Gym Lite 目前係計劃接入嘅外部題目來源，唔係現時呢批內部 benchmark tasks。

正式接入之前，仲要處理題目資料、真實 repository 同指定版本、每個 project 嘅執行依賴、測試對應，同埋 training 同 evaluation 題目嘅分隔。

## 收尾

我用幾句總結整個 pipeline：

第一，benchmark 提供要修正嘅題目。

第二，Agent 透過 Harness 同工具喺自己嘅 sandbox 入面檢查同修改程式。

第三，Verifier 用 private tests 評分，再將結果轉成 Reward。

第四，訓練程式整理 rollout、計 Advantage 同 clipped loss，然後更新 LoRA 參數。

目前我已經驗證過一次單步更新同 checkpoint 流程；至於模型能力有冇改善，仲需要進一步實驗證明。SWE-Gym Lite 則係未來計劃接入嘅題目來源。

> **一句講晒：我建立咗一條由出題、做題、改卷到嘗試更新模型嘅 pipeline，並逐步驗證每一段有冇接通。**
