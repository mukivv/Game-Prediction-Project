import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Game Prediction Project",page_icon="⭐", layout="wide")

@st.cache_resource
def load_assets():
    scaler_ml = joblib.load('models/ensemble/scaler.pkl')
    cols_ml = joblib.load('models/ensemble/model_columns.pkl')
    model_ml = joblib.load('models/ensemble/ensemble_model.pkl')
    
    model_nn = load_model('models/neural/game_nn_model.h5', compile=False)
    scaler_nn = joblib.load('models/neural/scaler_nn.pkl')
    cols_nn = joblib.load('models/neural/model_columns_nn.pkl')
    
    return model_ml, scaler_ml, cols_ml, model_nn, scaler_nn, cols_nn

model_ml, scaler_ml, cols_ml, model_nn, scaler_nn, cols_nn = load_assets()

st.markdown("""
    <style>
    /* 1. ซ่อนเฉพาะหัวข้อ Radio ใน Sidebar (Navigation) */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none;
    }
    
    /* 2. ซ่อนจุดวงกลมของ Radio ใน Sidebar */
    [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    /* 3. ปรับแต่งลักษณะของข้อความเมนู Sidebar */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 5px;
        cursor: pointer;
        display: flex;
        width: 100%;
        transition: all 0.3s;
        border: 1px solid transparent;
        background-color: rgba(255, 255, 255, 0.05);
    }

    /* 4. สไตล์เมื่อเอาเมาส์ไปวาง (Hover) */
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: rgba(151, 166, 195, 0.15);
        border: 1px solid rgba(151, 115, 223, 0.3);
    }

    /* 5. สไตล์เมื่อหน้าถูกเลือก (Selected) */
    [data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {
        background-color: rgba(151, 166, 195, 0.15) !important;
        color: white !important;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(151, 166, 195, 0.15);
    }

    /* 6. ตรวจสอบให้แน่ใจว่า Label ในหน้าหลัก (Main Content) แสดงผลปกติ */
    [data-testid="stMain"] [data-testid="stWidgetLabel"] {
        display: block !important;
        font-weight: bold;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* ปรับแต่งส่วน Container ของ Tabs ให้กว้างเต็มพื้นที่ */
    button[data-baseweb="tab"] {
        flex: 1;                /* บังคับให้แต่ละ Tab มีสัดส่วนเท่ากันและขยายเต็ม */
        text-align: center;      /* จัดข้อความให้อยู่ตรงกลาง */
    }

    /* ปรับแต่งเส้นใต้และระยะห่างของ Tabs */
    div[data-component="stTabs"] {
        width: 100%;
        padding-top: 10px;
    }

    /* ปรับฟอนต์และความสวยงามของตัวอักษรบน Tab */
    div[data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


st.sidebar.title("Game Prediction Project")
page = st.sidebar.radio("Navigation", [
    "ᯓ★ ML Description", 
    "ᯓ★ NN Description", 
    "ML ☆ Meta Score Prediction", 
    "NN ☆ Hit Classification"
])


if page == "ᯓ★ ML Description":
    st.title("📂 รายละเอียดการพัฒนาโมเดล Ensemble Learning")
    st.subheader("ᯓ★ แหล่งที่มาของข้อมูล (Dataset)")
    col1, col2 = st.columns(2)
    st.divider()
    st.subheader("ᯓ★ การเตรียมข้อมูล (Data Prepation)")
    col3, col4 = st.columns(2)
    with col1:
        st.write("- **Top Video Games 1995-2021 Metacritic**")
        st.write("- **URL :** https://www.kaggle.com/datasets/deepcontractor/top-video-games-19952021-metacritic")
        st.markdown("""
    | **Feature** | **คำอธิบาย** |
    | :--- | :--- |
    | `name` | ชื่อของเกม |
    | `platform` | เครื่องเล่นเกม (เช่น PC, PS4) |
    | `release_date` | วันเดือนปีที่เกมออกจำหน่าย |
    | `summary` | รายละเอียดหรือเนื้อเรื่องย่อของเกม |
    | `meta_score` | คะแนนวิจารณ์จากสื่อ |
    | `user_review` | คะแนนรีวิวจากผู้เล่นทั่วไป |
    """)
        st.write("> พัฒนาโดยใช้ข้อมูลจาก **Kaggle** และสนับสนุนการเขียนโค้ด/อธิบายทฤษฎีโดย **Google Gemini AI**")
    with col2:
        st.image("./image/ML_data.png", caption="ตัวอย่างข้อมูล Game Metacritic")
    with col3:
        st.subheader("Data Cleaning")
        st.markdown("""
    | **ขั้นตอน** | **คำอธิบาย** |
    | :--- | :--- |
    | Target Rows Cleaning | ลบแถวที่ข้อมูล `meta_score`ขาดหาย เนื่องจากเป็นค่าที่ต้องการข้อมูลจริงเพื่อให้โมเดลใช้เรียนรู้ |
    | Handling Missing Values | จัดการค่าว่างใน `summary` โดยใส่คำว่า "No summary" |
    | Data Type Conversion | แปลง `user_review` จาก string เป็น numeric และใช้ `errors='coerce'` เพื่อเปลี่ยน "tbd" เป็นค่าว่าง (NaN) |
    | Grouped Median/Mean Imputation | เติมค่าว่างด้วยค่าเฉลี่ยของ `user_review` ที่แยกตามกลุ่มคะแนน `meta_score` |
    """)
    with col4:
        st.subheader("Feature Engineering")
        st.markdown("""
    | **ขั้นตอน** | **คำอธิบาย** |
    | :--- | :--- |
    | Feature Extraction | ดึงข้อมูล `Year` และ `Month` ออกจากคอลัมน์ `release_date` เพื่อสร้าง Feature ใหม่ที่ช่วยให้โมเดลเข้าใจแนวโน้มตามเวลา - ปีที่วางจำหน่ายมีผลต่อมาตรฐานคะแนน และเดือนที่วางจำหน่ายอาจส่งผลต่อความนิยม |
    | One-Hot Encoding | ใช้ One-Hot Encoding ด้วย `pd.get_dummies` เพื่อเปลี่ยนข้อมูลใน `Platform` ให้กลายเป็นตัวเลขที่โมเดลคณิตศาสตร์เข้าใจได้ เนื่องจากอัลกอริทึม Linear Regression และ SVR ไม่สามารถประมวลผลข้อความได้โดยตรง |
    | Feature Scaling | แบ่งข้อมูล **Training 80% และ Testing 20%** จากนั้นใช้ StandardScaler ปรับช่วงของข้อมูลให้มีค่าเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1 ซึ่งจำเป็นสำหรับโมเดล SVR และ Linear Regression |
    """)
        

    st.divider()
    
    st.subheader("ᯓ★ ทฤษฎีของอัลกอริทึม (Ensemble Learning) ")

    st.info("""
**Ensemble Model** คือเทคนิคที่นำโมเดลพื้นฐาน (Base Learners) หลายตัวมาทำงานร่วมกัน [cite: 10] 
เพื่อรวมจุดแข็งและเฉลี่ยข้อผิดพลาดเข้าด้วยกัน ทำให้ได้ผลลัพธ์ที่มีความแม่นยำและเสถียร (Robustness) 
สูงกว่าการใช้โมเดลเพียงตัวเดียว
""")

    st.subheader("Voting Regressor Architecture")
    st.write("คือเทคนิคการเรียนรู้แบบกลุ่ม (Ensemble Learning) ที่ใช้หลักการหลายหัวดีกว่าหัวเดียว โดยการนำพยากรณ์จากโมเดลที่แตกต่างกันมาหา ค่าเฉลี่ย (Average) เพื่อรวมผลลัพธ์จาก 3 อัลกอริทึมหลัก:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🌲 Random Forest")
        st.caption("The Tree Expert")
        st.write("ใช้ Decision Trees หลายต้นเพื่อ**ลด Overfitting** โดยกำหนด `n_estimators=80` หมายถึงการสร้างต้นไม้ 80 ต้น แต่ละต้นจะสุ่มเลือกข้อมูลและ Features มาเรียนรู้แยกกัน จากนั้นจะนำผลพยากรณ์คะแนนจากต้นไม้ทั้ง 80 ต้นมาเฉลี่ยกัน")

    with col2:
        st.markdown("#### 📈 Linear Regression")
        st.caption("The Trend Finder")
        st.write("ใช้หาความสัมพันธ์เชิงเส้นเบื้องต้นของคะแนน เป็น**ฐานข้อมูลหลัก**ที่อ้างอิงตามแนวโน้มปกติของข้อมูล ไม่ให้โมเดลอื่นๆกระโดดไปมาจนเกินไป โดยโมเดลจะพยายามตีเส้นตรง (Linear Equation) ผ่านจุดข้อมูลเพื่อดูแนวโน้มว่า**เมื่อคะแนนรีวิวจากผู้ใช้ (User Review) สูงขึ้น คะแนนวิจารณ์ (Meta Score) จะมีแนวโน้มสูงขึ้นตามในสัดส่วนเท่าใด**")

    with col3:
        st.markdown("#### 🌌 SVR (RBF Kernel)")
        st.caption("The Kernel Master")
        st.write("เหมาะกับ**ข้อมูลที่มีตัวแปรหลายตัว** (High Dimensional Data) โดย SVR จะพยายาม**สร้างขอบเขต (Hyperplane)** ที่ครอบคลุมจุดข้อมูลให้ได้มากที่สุด โดย RBF Kernel จะช่วยแปลงข้อมูลจากมิติปกติให้กลายเป็นมิติที่สูงขึ้นเพื่อให้สามารถหาเส้นแบ่งความสัมพันธ์ที่ซับซ้อน (Non-linear) ได้แม่นยำขึ้น")

    st.divider()
    st.subheader("ᯓ★ ขั้นตอนการพัฒนาโมเดล (Model development steps)")

    tab1, tab2, tab3 = st.tabs(["1. Model Definition", "2. Ensemble Integration", "3. Evaluation"])
    
    with tab1:
        st.markdown("#### ᯓᡣ𐭩 การนิยามโมเดลย่อย (Defining Base Models)")
        st.write("โมเดลทั้ง 3 ประเภทถูกเลือกให้มีความหลากหลายของอัลกอริทึม (Model Diversity) เพื่อลดข้อผิดพลาดที่อาจเกิดขึ้นตามที่อธิบายไปในทฤษฎีของอัลกอริทึมด้านบน")
        st.code("""
        # 1. Random Forest (Tree-based)
            model1 = RandomForestRegressor(n_estimators=80, random_state=42)
        # 2. Linear Regression (Linear-based)
            model2 = LinearRegression()
        # 3. SVR (Kernel-based)
            model3 = SVR(kernel='rbf')""", language='python')

    with tab2:
        st.markdown("#### ᯓᡣ𐭩  การรวมโมเดล (Model Integration)")
        st.write("ใช้ **Voting Regressor** ทำหน้าที่มัดรวมโมเดลย่อยเข้าด้วยกัน และทำการฝึกสอน (Fit) ด้วยข้อมูลที่ผ่านการ Scale เรียบร้อยแล้ว (`X_train_scaled`)")
        st.code("""
        # สร้าง Ensemble Model (มัดรวมกัน)
            ensemble_model = VotingRegressor(estimators=[
                ('rf', model1),
                ('lr', model2),
                ('svr', model3)
            ])

        # สอนโมเดลด้วยข้อมูล Training Set
            ensemble_model.fit(X_train_scaled, y_train)""", language='python')

    with tab3:
        st.markdown("#### ᯓᡣ𐭩  การพยากรณ์และวัดผล (Prediction & Evaluation)")
        st.write("ใช้ข้อมูลชุดทดสอบ (Test Set) ที่โมเดลไม่เคยเห็นมาก่อน มาทำการพยากรณ์และเปรียบเทียบกับค่าจริง:")
        st.code("""
        # ทดสอบพยากรณ์จาก Test Set
            y_pred = ensemble_model.predict(X_test_scaled)
            
        # แสดงผลการวัดประสิทธิภาพ
            print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
            print(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")""", language='python')

    st.divider()

    st.subheader("📊 ประสิทธิภาพของโมเดล (Performance)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Mean Absolute Error (MAE)", "7.09", help="ค่าเฉลี่ยความผิดพลาดของคะแนน")
        st.write("Ensemble Model มีค่า MAE อยู่ที่ 7.09 ซึ่งในอุตสาหกรรมเกม คะแนนวิจารณ์ (Metascore) มีความลำเอียง (Bias) สูงมากจากตัวบุคคลเช่น เกมเดียวกัน สื่อหนึ่งอาจให้ 90 อีกสื่ออาจให้ 80 การที่โมเดลบีบความผิดพลาดเหลือเพียง ±7 คะแนน แสดงว่าโมเดลสามารถจับมาตรฐานกลางของแพลตฟอร์มและแนวเกมนั้นๆ ได้แม่นยำแล้ว")
    with c2:
        st.metric("R-squared Score", "0.40", help="ความสามารถในการอธิบายความผันแปรของข้อมูล")
        st.write("สาเหตุที่ค่า R-squared อยู่ที่ 0.40 เนื่องจากคะแนนวิจารณ์เกม (Metascore) ไม่ได้ขึ้นอยู่กับปัจจัยเชิงปริมาณเพียงอย่างเดียว แต่ยังมีปัจจัยเชิงคุณภาพ เช่น กราฟิก ระบบเนื้อเรื่อง และความเสถียรของตัวเกม ซึ่งเป็นข้อมูลที่ไม่ได้ระบุอยู่ใน Dataset อย่างไรก็ตาม ค่า MAE ที่ 7.09 ยืนยันว่าโมเดลมีความสามารถในการพยากรณ์ที่ใกล้เคียงกับมาตรฐานความเห็นของสื่อมวลชนส่วนใหญ่ในระดับที่น่าเชื่อถือ")
    st.markdown("---")
    st.markdown("""
<div style="text-align: center;">
    <p style="font-size: 0.9em; color: gray;">
        Developed by Thitaree Kaewsuwan | Student ID: 6704062610305
    </p>
</div>
""", unsafe_allow_html=True)

elif page == "ML ☆ Meta Score Prediction":
    st.title("🎮 ทดสอบการทำนายคะแนน Meta Score (ML)")
    
    available_platforms = [col.replace('platform_', '') for col in cols_ml if col.startswith('platform_')]
    available_platforms.sort()

    with st.form("prediction_form"):
        st.subheader("⸜(｡˃ ᵕ ˂)⸝♡ ระบุข้อมูลของเกมที่ต้องการพยากรณ์")
        col1, col2 = st.columns(2)
        
        with col1:
            u_rev = st.number_input("User Review Score ( 0.0 - 10.0 )", 0.0, 10.0, 7.5,)
            p_sel = st.selectbox("Platform", available_platforms)
            
        with col2:
            y_sel = st.number_input("Year of Release ( 1995 - 2026 )", 1995, 2026, 2024)
            m_sel = st.select_slider("Month of Release", options=list(range(1, 13)), value=6)
            
        submit_button = st.form_submit_button("Start Prediction")
    
    if submit_button:
        # เตรียมข้อมูล Input
        input_df = pd.DataFrame([[u_rev, y_sel, m_sel]], columns=['user_review', 'year', 'month'])
        for col in cols_ml:
            if col.startswith('platform_'): 
                input_df[col] = 1 if f"platform_{p_sel}" == col else 0
        
        final_input = scaler_ml.transform(input_df[cols_ml])
        res = model_ml.predict(final_input)
        
        if res[0] >= 80:
            st.success("**Must Play!** (เกมนี้มีโอกาสได้รับความนิยมสูงมาก)")
        elif res[0] >= 60:
            st.info("**Good** (เกมระดับมาตรฐาน อยู่ในเกณฑ์ดี)")
        else:
            st.warning("**Mixed** (คะแนนอยู่ในเกณฑ์ปานกลางหรือก้ำกึ่ง)")
        
        st.subheader("Predicted Meta Score")
        st.header(f"{res[0]:.2f}")
    st.markdown("---")
    st.markdown("""
<div style="text-align: center;">
    <p style="font-size: 0.9em; color: gray;">
        Developed by Thitaree Kaewsuwan | Student ID: 6704062610305
    </p>
</div>
""", unsafe_allow_html=True)

elif page == "ᯓ★ NN Description":
    st.title("🧠 รายละเอียดการพัฒนาโมเดล Neural Network")
    
    st.subheader("ᯓ★ แหล่งที่มาของข้อมูล (Dataset)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("- **Video Games Sales as at 22 Dec 2016**")
        st.write("- **URL:** https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings")
        st.image("./image/NN_data.png", caption="ตัวอย่างข้อมูล Game Sales")
        st.write("> พัฒนาโดยใช้ข้อมูลจาก **Kaggle** และสนับสนุนการเขียนโค้ด/อธิบายทฤษฎีโดย **Google Gemini AI**")

    with col2:
        st.markdown("""
| **Feature** | **คำอธิบาย** |
| :--- | :--- |
| `Name` | ชื่อของวิดีโอเกม |
| `Platform` | แพลตฟอร์มหรือเครื่องเล่นเกม (เช่น PC, PS4, Wii) |
| `Year_of_Release` | ปีที่เกมออกวางจำหน่าย |
| `Genre` | แนวเกม (เช่น Action, Sports, RPG) |
| `Publisher` | ชื่อบริษัทผู้จัดจำหน่ายเกม |
| `NA_Sales` | ยอดขายในอเมริกาเหนือ (หน่วย: ล้านชุด) |
| `EU_Sales` | ยอดขายในยุโรป (หน่วย: ล้านชุด) |
| `JP_Sales` | ยอดขายในญี่ปุ่น (หน่วย: ล้านชุด) |
| `Other_Sales` | ยอดขายในภูมิภาคอื่นๆ ทั่วโลก (หน่วย: ล้านชุด) |
| `Global_Sales` | ยอดขายรวมทั่วโลก (ใช้สำหรับคำนวณ Success Level) |
| `Critic_Score` | คะแนนวิจารณ์เฉลี่ยจากสื่อมวลชน (0-100) |
| `Critic_Count` | จำนวนสื่อที่ร่วมให้คะแนนวิจารณ์ |
| `User_Score` | คะแนนรีวิวเฉลี่ยจากผู้เล่นทั่วไป (0-10) |
| `User_Count` | จำนวนผู้เล่นที่ร่วมให้คะแนนรีวิว |
| `Developer` | บริษัทหรือทีมผู้พัฒนาเกม |
| `Rating` | การจัดระดับความเหมาะสมของผู้เล่น (เช่น E, T, M) |
""")

    st.divider()

    st.subheader("ᯓ★ การเตรียมข้อมูล (Data Preparation)")

    st.subheader("Data Cleaning")
    st.markdown("""
    | **ขั้นตอน** | **คำอธิบาย** |
    | :--- | :--- |
    | Data Type Conversion | แปลง `User_Score` และ `Year_of_Release` จาก string เป็นเลขและใช้ `errors='coerce'` เพื่อเปลี่ยน "tbd" และ "N/A" เป็นค่าว่าง (NaN) |
    | Handling Numeric NaN | เติมค่าว่างใน `Critic_Score` และ `User_Score` ด้วยค่า **Median แยกตามแนวเกม (Genre)** เพื่อรักษาเอกลักษณ์ของแต่ละแนวเกม ส่วนค่าที่คิดตามหลักการข้างต้นไม่ได้จะนำค่าเฉลี่ยของ Feature นั้นๆ มาใส่แทน |
    | Handling Missing Values | เติมค่าว่างใน `Year_of_Release` ด้วย**ค่าฐานนิยม** คือเอาปีที่เกมออกมากที่สุดมาใส่
    | Categorical Cleaning | เติมค่าว่างในคอลัมน์ที่มี Data type เป็น String (เช่น Publisher, Developer) ด้วยคำว่า **'Unknown'**  |
    | Labeling Success | สร้าง Target `success_level` (0-3) โดยแบ่งจากยอดขาย: **AAA Hit (>10M)**, **Successful (>1M)**, **Mid-range (>0.1M)** และ **Low-sales**  |
    """)

    st.subheader("Feature Engineering")
    st.markdown("""
    | **ขั้นตอน** | **คำอธิบาย** |
    | :--- | :--- |
    | One-Hot Encoding | ใช้ `pd.get_dummies` แปลง `Genre` และ `Platform` เป็นตัวเลขเพื่อให้ Neural Network เรียนรู้ความสัมพันธ์เชิงลึกได้  |
    | Feature Scaling | แบ่งข้อมูล **Training 80% และ Testing 20%** จากนั้นใช้ **StandardScaler** ปรับข้อมูลให้มีค่าเฉลี่ยเป็น 0 เพื่อให้โมเดลประมวลผล Weight ได้รวดเร็วและไม่เอนเอียง  |
    """)

    st.divider()

    st.subheader("ᯓ★ ทฤษฎีโครงสร้างโมเดล (Neural Network Architecture)")
    st.info("""
    โมเดลที่ใช้ในโปรเจกต์นี้คือ Multi-Layer Perceptron (MLP) ซึ่งเป็น Feedforward Neural Network ที่ประกอบด้วยเลเยอร์ซ้อนทับกัน ซึ่งประกอบด้วยหน่วยประมวลผลขนาดเล็กที่เรียกว่า Neuron วางตัวเรียงกันเป็นชั้นๆ อย่างน้อย 3 ชั้น (layers) เพื่อเรียนรู้รูปแบบยอดขายเกมที่ซับซ้อน
    """)
   
    st.markdown("""
    #### **1. โครงสร้างเลเยอร์ (Layer Hierarchy)**
* **Input Layer**: รับข้อมูลนำเข้า (Features) ที่ผ่านการทำ One-Hot Encoding เช่น Platform และ Genre
* **Hidden Layers (3 Layers)**: ทำหน้าที่สกัดคุณลักษณะ (Feature Extraction) เพื่อหาความสัมพันธ์เชิงลึก
    * **Layer 1 (256 Nodes)**: เรียนรู้ความสัมพันธ์พื้นฐานของข้อมูลในมิติที่กว้าง
    * **Layer 2 (128 Nodes)**: คัดกรองและประมวลผลข้อมูลต่อจากชั้นแรกให้มีความจำเพาะเจาะจงมากขึ้น
    * **Layer 3 (64 Nodes)**: บีบอัดข้อมูลให้เหลือเพียงสาระสำคัญก่อนส่งไปยังส่วนการจำแนกผล
* **Output Layer (4 Nodes)**: พยากรณ์ความน่าจะเป็นของระดับความสำเร็จทั้ง 4 คลาส (Low-sales จนถึง AAA Hit)

#### **2. กลไกการประมวลผล (Processing Mechanics)**
* **Weights & Bias**: ค่าน้ำหนักและค่าเบี่ยงเบนจะถูกปรับเปลี่ยนตลอดช่วงการฝึกสอนผ่านกระบวนการ **Backpropagation** เพื่อลดค่าความผิดพลาดให้เหลือน้อยที่สุด
* **Activation Function (ReLU)**: ใช้ในชั้นซ่อนเพื่อเพิ่มคุณสมบัติความไม่เป็นเส้นตรง (**Non-linearity**) ช่วยให้โมเดลแก้ปัญหาที่ซับซ้อนกว่าสมการเส้นตรงปกติได้
* **Output Activation (Softmax)**: ใช้ในชั้นสุดท้ายเพื่อแปลงค่าที่คำนวณได้ให้เป็น "ความน่าจะเป็น" (Probability) โดยผลรวมของทุกคลาสจะเท่ากับ 1.0 เสมอ

#### **3. การควบคุมประสิทธิภาพและการหาจุดเหมาะสม (Optimization)**
* **Dropout (0.2 - 0.3)**: เทคนิคการสุ่ม "ปิด" โหนดบางส่วน เพื่อป้องกันปัญหา **Overfitting** และบังคับให้โครงข่ายเรียนรู้คุณลักษณะที่หลากหลาย
* **Adam Optimizer**: อัลกอริทึมที่ปรับอัตราการเรียนรู้ (**Learning Rate**) แบบอัตโนมัติ ช่วยให้โมเดลเข้าสู่จุดสมดุล (Convergence) ได้รวดเร็ว
* **Loss Function**: วัดผลด้วย **Sparse Categorical Crossentropy** ซึ่งออกแบบมาสำหรับการจำแนกหลายหมวดหมู่โดยเฉพาะ
""")

    st.divider()
    st.subheader("ᯓ★ ขั้นตอนการพัฒนาโมเดล (Development Steps)")
    
    tab1, tab2, tab3 = st.tabs(["1. Model Building", "2. Model Training", "3. Evaluation"])

    with tab1:
        st.markdown("#### 🛠️ การสร้างโครงสร้างโมเดล (Architecture)")
        st.write("นิยามโครงสร้าง **Neural Network (MLP)** แบบลำดับ โดยมีการใช้ Dropout เพื่อป้องกันปัญหา Overfitting และฟังก์ชันกระตุ้นที่เหมาะสมในแต่ละเลเยอร์")
        st.code("""
        # สร้างโครงสร้างโมเดลแบบ Sequential (เรียงต่อกัน)
            model_nn = Sequential([
            # Layer 1: รับข้อมูล Input และขยายผลเป็น 256 Nodes
                Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dropout(0.3), # สุ่มปิดโหนด 30% เพื่อคุม Overfitting

            # Layer 2: ชั้นซ่อนขนาด 128 Nodes
                Dense(128, activation='relu'),
                Dropout(0.2), # สุ่มปิดโหนด 20%

            # Layer 3: ชั้นซ่อนขนาด 64 Nodes เพื่อบีบอัดข้อมูล
                Dense(64, activation='relu'),

            # Layer Output: 4 Nodes พยากรณ์ความน่าจะเป็น 4 คลาส (Softmax)
                Dense(4, activation='softmax')
            ])""", language='python')

    with tab2:
        st.markdown("#### ⚙️ การตั้งค่าและการเทรน (Training)")
        st.write("กำหนดกระบวนการเรียนรู้ด้วย **Adam Optimizer** และทำการฝึกสอน (Fit) โมเดลโดยแบ่งข้อมูลส่วนหนึ่งไว้สำหรับประเมินผลระหว่างเทรน")
        st.write("- **Sparse Categorical Crossentropy** คือ ฟังก์ชันสูญเสีย (Loss Function) ที่ใช้สำหรับโมเดลการจำแนกประเภท (Classification) ที่มีผลลัพธ์มากกว่า 2 กลุ่มขึ้นไป (Multi-class) โดยมีหน้าที่หลักคือการวัด**ระยะห่าง**ระหว่างค่าที่โมเดลพยากรณ์ออกมากับค่าจริง เพื่อให้โมเดลปรับตัวให้แม่นยำขึ้นในรอบถัดไป")
        st.code("""
        # การตั้งค่าการเรียนรู้ (Compile Configuration) 
        # กำหนด Adam Optimizer พร้อม Learning Rate 0.001 (ความเร็วในการปรับตัวของโมเดล)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 

            model_nn.compile(
                optimizer=optimizer,
            # ใช้ Sparse Categorical Crossentropy เพื่อวัดความผิดพลาดของการจำแนก 4 คลาส
                loss='sparse_categorical_crossentropy', 
            # ใช้ Accuracy เป็นตัวชี้วัดหลักในการประเมินความแม่นยำ
                metrics=['accuracy']
            )

        # ฝึกสอนโมเดล ( Model Fitting )
            history = model_nn.fit(
                X_train_scaled, y_train,
            # Epochs: กำหนดจำนวนรอบการเรียนรู้ทั้งหมด 60 รอบ
                epochs=60,          
            # Batch Size: จำนวนข้อมูลที่ใช้คำนวณก่อนปรับ Weight ในแต่ละครั้ง (ช่วยเรื่องความเสถียร)
                batch_size=64,      
            # Validation Split: แบ่งข้อมูล 20% ออกมาเพื่อทดสอบความแม่นยำระหว่างเทรน (ป้องกัน Overfitting)
                validation_split=0.2, 
            # Verbose: แสดงสถานะการเทรนในแต่ละรอบ
                verbose=1
            )""", language='python')

    with tab3:
        st.markdown("#### 📊 การวัดผลประสิทธิภาพ (Evaluation)")
        st.write("ใช้ข้อมูลชุดทดสอบ (Test Set) ที่โมเดลไม่เคยเห็นมาก่อน มาทำการประเมินค่าความแม่นยำ (Accuracy) เพื่อวัดผลการใช้งานจริง")
        st.code("""
        # 1. ทดสอบประเมินผลด้วยข้อมูลชุด Test Set
            loss, accuracy = model_nn.evaluate(X_test_scaled, y_test)
            
        # 2. แสดงผลค่าความแม่นยำรวมของโมเดล
            print(f"Neural Network Accuracy: {accuracy*100:.2f}%")""", language='python')

    st.divider()

    st.subheader("📊 ประสิทธิภาพของโมเดล (Performance)")
    c1, c2 = st.columns(2)
    
    with c1:
        st.metric("Model Accuracy", "60.26%", help="สัดส่วนความถูกต้องในการพยากรณ์จากข้อมูลชุดทดสอบ")
        st.write("""
        **ความแม่นยำ (Accuracy):** อยู่ที่ **60.26%** ซึ่งอยู่ในเกณฑ์ที่น่าพอใจสำหรับ Dataset ประเภทยอดขาย (Market Success) 
        ที่มีความผันผวนสูงจากปัจจัยภายนอก (เช่นกระแสไวรัล คู่แข่ง) โดยโมเดลมีความสามารถโดดเด่นในการจำแนกความแตกต่างระหว่าง 
        **'Low-sales'** และ **'AAA Hit'** ได้อย่างมีนัยสำคัญ
        """)
    
    with c2:
        st.metric("Final Loss", "0.84", help="ค่าความผิดพลาดสะสมหลังจบกระบวนการฝึกสอน")
        st.write("""
        **ค่าความสูญเสีย (Loss):** ลดลงอย่างต่อเนื่องจนคงที่ที่ **0.84** บ่งบอกถึงแนวโน้มการเรียนรู้ (**Learning Curve**) 
        ที่เสถียร การใช้เทคนิค **Dropout** ช่วยควบคุมไม่ให้เกิดปัญหา **Overfitting** รุนแรง 
        ส่งผลให้โมเดลมีประสิทธิภาพในการปรับตัว (Generalization) เพื่อทำนายข้อมูลเกมใหม่ๆ ได้อย่างยืดหยุ่น
        """)
    st.markdown("---")
    st.markdown("""
<div style="text-align: center;">
    <p style="font-size: 0.9em; color: gray;">
        Developed by Thitaree Kaewsuwan | Student ID: 6704062610305
    </p>
</div>
""", unsafe_allow_html=True)
        
elif page == "NN ☆ Hit Classification":
    st.title("🎯 ทดสอบทำนายระดับความสำเร็จ (Neural Network)")
    
    available_genres = sorted([col.replace('Genre_', '') for col in cols_nn if col.startswith('Genre_')])
    available_platforms = sorted([col.replace('Platform_', '') for col in cols_nn if col.startswith('Platform_')])

    with st.form("nn_prediction_form"):
        st.subheader("⸜(｡˃ ᵕ ˂ )⸝♡ ระบุข้อมูลของเกมที่ต้องการพยากรณ์")
        col1, col2 = st.columns(2)
        
        with col1:
            g_sel = st.selectbox("Genre", available_genres)
            p_sel = st.selectbox("Platform Selection", available_platforms)
            year_s = st.number_input("Release Year ( 1980 - 2026 )", 1980, 2026, 2015)
            
        with col2:
            user_s = st.slider("User Score", 0.0, 10.0, 7.0)
            crit_s = st.slider("Critic Score", 0, 100, 70)
            
        submit_button = st.form_submit_button("Start Prediction")

    if submit_button:
        input_nn = pd.DataFrame([[crit_s, user_s, year_s]], 
                                columns=['Critic_Score', 'User_Score', 'Year_of_Release'])
        
        for col in cols_nn:
            if col.startswith('Genre_'):
                input_nn[col] = 1 if f"Genre_{g_sel}" == col else 0
            elif col.startswith('Platform_'):
                input_nn[col] = 1 if f"Platform_{p_sel}" == col else 0
        
        input_nn = input_nn[cols_nn]
        
        final_nn = scaler_nn.transform(input_nn)
        pred_prob = model_nn.predict(final_nn)
        
        classes = ['Low-sales', 'Mid-range', 'Successful', 'AAA Hit']
        result = classes[np.argmax(pred_prob)]
        
        st.toast("ทำนายผลสำเร็จ ✅️")

        if result == 'AAA Hit':
            st.success(f"**{result}**: เกมนี้มีโอกาสเป็นระดับตำนาน มียอดขายมากกว่า 10 ล้านชุด")
        elif result == 'Successful':
            st.info(f"**{result}**: เกมประสบความสำเร็จตามมาตรฐาน มียอดขายที่น่าพึงพอใจ 1 - 10 ล้านชุด")
        elif result == 'Mid-range':
            st.warning(f"**{result}**: เกมอยู่ในกลุ่มที่มียอดขายปานกลาง 0.1 - 1 ล้านชุด")
        else:
            st.error(f"**{result}**: เกมอยู่ในกลุ่มที่มียอดขายค่อนข้างน้อย ซึ่งน้อยกว่า 0.1 ล้านชุด")

        st.subheader("Predicted Success Level")
        st.header(f"{result}")
    st.markdown("---")
    st.markdown("""
<div style="text-align: center;">
    <p style="font-size: 0.9em; color: gray;">
        Developed by Thitaree Kaewsuwan | Student ID: 6704062610305
    </p>
</div>
""", unsafe_allow_html=True)