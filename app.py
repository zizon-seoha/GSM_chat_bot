import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# 페이지 기본 설정
st.set_page_config(page_title="GSM 선배 챗봇", page_icon="🧑‍🎓")
st.title("🧑‍🎓 GSM 길잡이 선배에게 물어봐!")

# 구글 API 키를 스트림릿 비밀 저장소에서 가져오기 (배포할 때 필요함)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 1. 데이터 로드 및 벡터 DB 구축 (캐싱하여 속도 향상!)
@st.cache_resource
def load_data_and_db():
    # 주의: CSV 파일 이름이 실제 파일명과 똑같아야 합니다!
    df = pd.read_csv("GSM 길잡이 AI 챗봇을 위한 데이터(응답) - 설문지 응답 시트1 (8).csv")
    
    # 불필요한 데이터 삭제
    cols_to_drop = ['타임스탬프', '이메일 주소', '더 하고 싶은 말이 있으시다면 카테고리를 클릭하고 \n작성해주십쇼!']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 빈칸 제외하고 텍스트로 합치기
    texts = []
    for _, row in df.iterrows():
        row_data = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        texts.append(" | ".join(row_data))
        
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = load_data_and_db()

# 2. AI 모델 및 프롬프트 설정
llm = ChatGoogleGenerativeAI(model="Gemini-3.1-Flash-Lite", temperature=0.5)

system_prompt = """당신은 광주소프트웨어마이스터고(GSM)를 너무나 사랑하는 '유쾌하고 따뜻한 3학년 선배'입니다.
후배가 질문을 하면, 아래 [참고 정보]를 꼼꼼히 읽은 뒤 **완벽하게 소화해서 선배의 언어로 재구성해** 대답해 주세요.

[절대 지켜야 할 답변 규칙] 🚨
1. 기계적인 나열 절대 금지: "정보에 따르면~", "다음과 같습니다.", "1번, 2번" 처럼 딱딱하게 번호를 매기며 로봇처럼 읽어주지 마세요.
2. 자연스러운 스토리텔링: [참고 정보]의 문장들을 그대로 복사+붙여넣기 하지 마세요. 여러 선배들의 꿀팁을 하나로 자연스럽게 엮어서 "내가 경험해 보니까 이렇더라~" 하는 식으로 썰을 풀듯이 말해주세요.
3. 완벽한 구어체 사용: "안녕 후배야!", "이건 진짜 꿀팁인데~", "다들 화이팅하자!" 처럼 친한 동네 형/누나/언니/오빠 같은 말투(반말과 해요체를 섞어서)를 사용하세요.
4. 개인정보 차단: 이름, 이메일 등은 절대 언급하지 마세요.
5. 모르는 질문 대처: 정보에 없는 내용을 물어보면 지어내지 말고 "앗, 그건 나도 잘 모르겠어! 다른 선배나 선생님께 여쭤보는 게 좋겠다ㅎㅎ"라고 쿨하게 넘기세요.

[참고 정보]:
{context}"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# 3. 채팅 화면 구현
# 대화 기록을 저장할 공간 만들기
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕 후배야! 학교 생활에 대해 궁금한 거 있어?"}]

# 이전 대화 내용 화면에 그리기
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요 (예: 1학년 때 뭐부터 해야 해?)")

if user_input:
    # 내 질문 화면에 띄우고 기록
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        try:
            # 스피너(로딩) 대신 곧바로 타이핑을 시작하도록 바꿉니다!
            response_stream = rag_chain.stream(user_input) # 실시간으로 쪼개서 가져오기
            
            # st.write_stream이 타다다닥 치는 효과를 자동으로 만들어줍니다.
            response = st.write_stream(response_stream) 
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error("앗! 지금 나한테 질문하는 후배들이 너무 많아서 숨이 차네 헥헥 💦 딱 1분만 이따가 다시 질문해줄래?")
