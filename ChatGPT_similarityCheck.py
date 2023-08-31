#Coded By Haosen Yan 30.05.2023,update on 02.06.2023,update 07.06.2023
#Only for Project Factor check(experimental)
#Use three kinds of Cosin's Model for similarity check (could work for Multi language)
import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from text2vec import Similarity
import sys
sys.path.append('..')



def main():
    st.title("Factor Checker")
    st.write("ChatGPT Reliability Comparison Tool")
    st.write("Coded by Haosen Yan only for ChatGPT Project (experimental)")
    st.write("Could work with multilanguage ")

    sentence1 = st.text_area("Enter ChatGPT Info")
    sentence2 = st.text_area("Enter reliable resource")

    if st.button("Check"):
        st.write("Calculating...")
        # 显示调皮的文字表情
        st.markdown('''
            <div style="font-size: 24px; text-align: center;">
                <span style="color: red;">😜</span>Wait a moment<span style="color: red;">😜</span>
            </div>
        ''', unsafe_allow_html=True)
        # Calculate embedding-vector Model
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        embeddings1 = model.encode([sentence1])
        embeddings2 = model.encode([sentence2])
        similarity_score = util.dot_score(embeddings1, embeddings2)[0]
        embedding_score = similarity_score.item()

        # Cosin Model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        #Compute embedding for both lists
        embeddings1 = model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model.encode(sentence2, convert_to_tensor=True)
        #Compute cosine-similarities
        cosine_score = util.cos_sim(embeddings1, embeddings2)
        cosine_scores = cosine_score.item()
        
        # semantic search
        sim_model = Similarity()
        score = sim_model.get_score(sentence1, sentence2)
      

        st.write("Embedding-vector:", round(embedding_score,4))
        st.write("CoSin:", round(cosine_scores,4))
        st.write("Semantic search:", round(score,4))

        # 获取数据
        embedding_percentage = embedding_score * 100
        cosine_percentage = cosine_scores * 100
        semantic_percentage = score * 100



        # 创建三个饼图
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        # 饼图1：Embedded-Vector
        # ax1.pie([embedding_percentage, 100-embedding_percentage], labels=['', ''], autopct='%.4f%%', startangle=90)
        # ax1.axis('equal')
        # ax1.set_title('Embedded-Vector')

        # 饼图2：Cosine Similarity
        # ax2.pie([cosine_percentage, 100-cosine_percentage], labels=['', ''], autopct='%.4f%%', startangle=90)
        # ax2.axis('equal')
        # ax2.set_title('Cosine Similarity')

        # 饼图3：Semantic Search
        # ax3.pie([semantic_percentage, 100-semantic_percentage], labels=['', ''], autopct='%.4f%%', startangle=90)
        # ax3.axis('equal')
        # ax3.set_title('Semantic Search')

        # 调整子图间的间距
        # plt.subplots_adjust(wspace=0.4)

        # 显示图表
        # st.pyplot(fig)

        # Plotting bar chart
        labels = ['embedding-vector','CoSin','semantic search']
        values = [embedding_score, cosine_scores, score]
        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel('Check Score')
        ax.set_title('Comparison')

        st.pyplot(fig)

        # import numpy as np
        #
        # labels = ['embedding-vector', 'CoSin', 'semantic search']
        # values = [embedding_score, cosine_scores, score]
        #
        # fig, ax = plt.subplots()
        # bars = ax.bar(labels, values)
        # ax.set_ylabel('Check Score')
        # ax.set_title('Comparison')
        #
        # threshold = 0.85  # 设定阈值
        #
        # for bar in bars:
        #     if bar.get_height() < threshold:
        #         bar.set_height()  # 将比例低于阈值的柱子高度设置为0.5
        #
        # st.pyplot(fig)


if __name__ == '__main__':
    main()

##to run this code you should install text2vec, SentenceTransformer
##use code streamlit run XXXX.py