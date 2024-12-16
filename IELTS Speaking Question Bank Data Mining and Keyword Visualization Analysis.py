# -*- coding: utf-8 -*-
"""
IELTS 口語題庫數據挖掘與關鍵字可視化分析
此程式分析 PDF 中的關鍵詞，結合 nltk 停用詞與外部檔案的停用詞，並生成詞雲圖。
"""

import re
import pdfplumber  # 用於提取 PDF 內容
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF 演算法
from collections import Counter  # 詞頻統計
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

# 確保 nltk 的停用詞資源已下載
nltk.download('stopwords')

# --- 載入停用詞（nltk + 外部檔案）---
def load_stopwords(file_path=None):
    """
    載入停用詞，結合 nltk 提供的停用詞與外部檔案的停用詞
    :param file_path: 外部停用詞檔案路徑，可選，若提供則額外加入外部的停用詞
    :return: 停用詞集合
    """
    stopwords_set = set(stopwords.words('english'))  # 獲取 nltk 的內建英文停用詞集合

    if file_path:  # 如果有提供外部檔案路徑
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                extra_stopwords = set(f.read().splitlines())  # 讀取外部停用詞並轉為集合
            stopwords_set.update(extra_stopwords)  # # 合併 nltk 與外部停用詞到停用詞集合中
        except FileNotFoundError:  # 若檔案不存在則給出提示
            print(f"外部停用詞檔案 '{file_path}' 不存在，將僅使用 nltk 提供的停用詞。")
    return stopwords_set

# --- 清理文字內容 ---
def clean_text(text, stopwords):
    """
    清理文字內容並移除停用詞
    :param text: 原始文本
    :param stopwords: 停用詞集合
    :return: 清理後的文字內容（移除停用詞後的結果）
    """
    # 移除非字母的字符（保留空白符以便分詞）
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 將文本轉換為小寫並分詞
    words = text.lower().split()
    # 移除停用詞
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)  # 將處理後的單詞重新組合為字串

# --- 讀取 PDF 檔案內容 ---
def extract_text_from_pdf(file_path):
    """
    從 PDF 檔案中提取所有頁的文字內容
    :param file_path: PDF 檔案路徑
    :return: 提取的所有文字內容
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:  # 開啟 PDF 檔案
        for page in pdf.pages:  # 遍歷每一頁
            text += page.extract_text()  # 提取文字並累加
    return text

# --- 主程式 ---
# 1. 提取 PDF 內容
pdf_file = '【完整版口语题库-题目版】2024年9-12月口语题库-新.pdf'  # 指定 PDF 檔案路徑
all_text = extract_text_from_pdf(pdf_file)  # 提取 PDF 內容

# 2. 載入停用詞
stopwords_file = 'adjustable-english.txt'  # 指定外部停用詞檔案名稱
stopwords_set = load_stopwords(file_path=stopwords_file)  # 合併外部與 nltk 的停用詞

# 3. 清理文字內容
cleaned_text = clean_text(all_text, stopwords_set)  # 清理 PDF 中的文字內容

# 4. 使用 TF-IDF 提取關鍵字
def extract_keywords_tfidf(text, top_k=100):
    """
    使用 TF-IDF 方法提取關鍵字
    :param text: 清理後的文本
    :param top_k: 提取的關鍵字數量
    :return: 關鍵字及其 TF-IDF 分數的列表
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])  # 計算 TF-IDF 矩陣
    feature_names = vectorizer.get_feature_names_out()  # 獲取特徵名稱（關鍵字）
    scores = tfidf_matrix.toarray()[0]  # 獲取 TF-IDF 分數
    keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return keywords

keywords_top = extract_keywords_tfidf(cleaned_text, top_k=100)  # 提取 TF-IDF 關鍵字
print("關鍵詞（TF-IDF）前 100 名：")
for word, weight in keywords_top:
    print(f"{word}: {weight}")

# 5. 詞頻統計
def word_frequency(text):
    """
    統計文本中每個單詞的出現次數
    :param text: 清理後的文本
    :return: 單詞及其出現次數的字典
    """
    words = text.split()
    return Counter(words)  # 返回詞頻統計結果

freq_results = word_frequency(cleaned_text)  # 統計詞頻

# 6. 將詞頻結果輸出為 CSV 檔案
csv_file = 'IELTS_SPEAK關鍵詞詞頻分析_英文.csv'  # 指定輸出的 CSV 檔案名稱
df_result = pd.DataFrame(freq_results.items(), columns=['關鍵詞語', '出現次數'])  # 將詞頻結果轉為 DataFrame
df_result = df_result.sort_values(by='出現次數', ascending=False)  # 按出現次數降序排序
df_result.to_csv(csv_file, encoding='utf-8-sig', index=False)  # 輸出為 CSV
print(f"關鍵詞詞頻分析結果已儲存至 '{csv_file}'。")

# 7. 生成詞雲圖
keywords_data = pd.read_csv(csv_file, encoding='utf-8-sig')  # 從 CSV 讀取資料
wordcloud = WordCloud(
    background_color='white',  # 設定背景為白色
    width=800,  # 圖片寬度
    height=400,  # 圖片高度
    max_words=100  # 最大關鍵詞數量
)

# 生成詞雲所需的詞頻字典
word_freq = dict(zip(keywords_data['關鍵詞語'], keywords_data['出現次數']))
wordcloud.generate_from_frequencies(word_freq)  # 根據詞頻生成詞雲

# 繪製並保存詞雲圖
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 不顯示座標軸
plt.title('IELTS_SPEAK Word Cloud', fontsize=16)  # 設定標題
plt.savefig('IELTS_SPEAK_WordCloud詞雲圖.png')  # 保存詞雲圖為 PNG 檔案
plt.show()

print("詞雲圖已生成並保存為 'IELTS_SPEAK_WordCloud詞雲圖.png'！")
