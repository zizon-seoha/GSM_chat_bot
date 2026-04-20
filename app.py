import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="GSM 선배 챗봇", page_icon="🧑‍🎓", layout="centered")

# 커스텀 CSS로 글꼴 크기 및 간격 조정 (가독성 향상)
st.markdown("""
    <style>
    .main { font-size: 1.1rem; }
    .stChatMessage { margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧑‍🎓 GSM 길잡이 선배")
st.subheader("학교 생활, 프로젝트, 진로 고민까지! 선배가 다 알려줄게.")

# 사이드바에 안내 문구 추가 (메인 화면을 깨끗하게 유지)
with st.sidebar:
    st.image("https://images.velog.io/images/im_chang_1217/post/87073cce-cbea-4079-af78-cca389b17f7a/GSM.png", width=100) # 학교 로고가 있다면 주소 입력
    st.title("📌 이용 안내")
    st.info("""
    - **질문 예시**
    - 1학년 때 뭐부터 하면 좋아?
    - 기숙사 생활 꿀팁 알려줘!
    - 프로젝트 할 때 팀장이면 어떡해?
    - 전공 선택 고민이야...
    """)
    st.warning("⚠️ 답변은 선배들의 설문 데이터를 바탕으로 AI가 생성한 것이니 참고용으로만 활용해줘!")

# API 키 설정
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 2. 데이터 로드 및 벡터 DB (캐싱)
@st.cache_resource
def load_data_and_db():
    try:
        df = pd.read_csv("GSM 길잡이 AI 챗봇을 위한 데이터(응답) - 설문지 응답 시트1 (8).csv")
        cols_to_drop = ['타임스탬프', '이메일 주소', '더 하고 싶은 말이 있으시다면 카테고리를 클릭하고 \n작성해주십쇼!']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        texts = []
        for _, row in df.iterrows():
            row_data = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            texts.append(" | ".join(row_data))
            
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        vectorstore = FAISS.from_texts(texts, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했어: {e}")
        return None

retriever = load_data_and_db()

# 3. AI 모델 및 가독성 특화 프롬프트
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.6)

system_prompt = """당신은 광주소프트웨어마이스터고(GSM)를 너무나 사랑하는 '유쾌하고 따뜻한 3학년 선배'입니다.
후배가 질문을 하면, 아래 [참고 정보]를 꼼꼼히 읽은 뒤 **완벽하게 소화해서 선배의 언어로 재구성해** 대답해 주세요.

[절대 지켜야 할 답변 규칙] 🚨
1. 기계적인 나열 절대 금지: "정보에 따르면~", "다음과 같습니다.", "1번, 2번" 처럼 딱딱하게 번호를 매기며 로봇처럼 읽어주지 마세요.
2. 자연스러운 스토리텔링: [참고 정보]의 문장들을 그대로 복사+붙여넣기 하지 마세요. 여러 선배들의 꿀팁을 하나로 자연스럽게 엮어서 "내가 경험해 보니까 이렇더라~" 하는 식으로 썰을 풀듯이 말해주세요.
3. 완벽한 구어체 사용: "안녕 후배야!", "이건 진짜 꿀팁인데~", "다들 화이팅하자!" 처럼 친한 동네 형/누나/언니/오빠 같은 말투(반말과 해요체를 섞어서)를 사용하세요.
4. 개인정보 차단: 이름, 이메일 등은 절대 언급하지 마세요.
5. 모르는 질문 대처: 정보에 없는 내용을 물어보면 지어내지 말고 "앗, 그건 나도 잘 모르겠어! 다른 선배나 선생님께 여쭤보는 게 좋겠다ㅎㅎ"라고 쿨하게 넘기세요.
6. **핵심 요약**: 답변 시작 부분에 한 줄로 핵심 요약을 해주세요.
7. **가독성 강조**: 중요한 단어나 문구는 **굵게(Bold)** 표시하세요.
8. **적절한 줄바꿈**: 문장이 너무 길어지지 않게 엔터(줄바꿈)를 자주 사용하세요.
9. **이모지 활용**: 친근감을 위해 문장 끝에 적절한 이모지를 사용하세요.
10. **구조화**: 내용이 많으면 '첫째, 둘째' 또는 '먼저, 그 다음은' 등의 표현을 써서 흐름을 만드세요.
11. 관련성 : 질문에 관련 있는 내용만 답하고 나머지 팁은 추가 적인 팁으로 표현하거나 없애세요.
12. **질문 우선순위**: 후배가 '백엔드', '공부법' 등 특정 주제를 물어보면 [참고 정보]에서 **그 주제와 직접 관련된 내용**을 최우선으로 찾아 답변하세요. 
13. **불필요한 조언 금지**: 질문과 상관없는 '인간관계', '선배와 친해지기' 등의 일반적인 조언은 [참고 정보]에 해당 내용이 메인이 아니라면 언급하지 마세요.
14. **구체적 수치/방법**: 데이터에 공부 사이트, 언어, 프레임워크 등이 있다면 생략하지 말고 정확하게 전달하세요.

[참고 정보]:
{context}"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# 4. 채팅 구현
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕 후배야! 😉 학교 생활에 대해 궁금한 게 있다면 편하게 물어봐. 선배가 아는 선에서 다 알려줄게!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("선배에게 질문하기...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        try:
            response_stream = rag_chain.stream(user_input)
            response = st.write_stream(response_stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error("앗, 지금 질문이 너무 많아서 대답이 지연되고 있어! 잠시 후에 다시 시도해줘. 🙏")
