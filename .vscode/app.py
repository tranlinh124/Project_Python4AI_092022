import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import re


st.title("BẢNG ĐIỂM  LỚP PY4AI 09/2022")
df = pd.read_csv("D:\AI\Project_Python4AI_092022\py4ai-score.csv")
tab1, tab2, tab3, tab4 = st.tabs(["Danh sách", "Biểu đồ", "Phân nhóm", "Phân loại"])

def clean_data():
    df['S1'].fillna(0, inplace = True)
    df['S2'].fillna(0, inplace = True)
    df['S3'].fillna(0, inplace = True)
    df['S4'].fillna(0, inplace = True)
    df['S5'].fillna(0, inplace = True)
    df['S6'].fillna(0, inplace = True)
    df['S7'].fillna(0, inplace = True)
    df['S8'].fillna(0, inplace = True)
    df['S9'].fillna(0, inplace = True)
    df['S10'].fillna(0, inplace = True)

with tab1:
    col1, col2, col3 ,col4, col5 = st.columns(5)
    with col1:
        def choose_gender():
            st.write("Giới tính")
            gender = ""
            check_nam = st.checkbox('Nam')
            check_nu = st.checkbox('Nữ')
            if (check_nam is True) and (check_nu is False):
                gender = "M"
            elif (check_nam is True) and (check_nu is True):
                gender = ""
            elif (check_nam is False) and (check_nu is True):
                gender = "F"
            
            return gender
        gender = choose_gender()
        # st.write(choose_gender())
        
    with col2:
        def choose_class_grade():
            class_grade = st.radio('Khối lớp', ('Khối 10', 'Khối 11', 'Khối 12', 'Tất cả'), horizontal=True)
            if class_grade == 'Khối 10':
                class_grade_chosen = '10'
            elif class_grade == 'Khối 11':
                class_grade_chosen = '11'
            elif class_grade == 'Khối 12':
                class_grade_chosen = '12'
            else:
                class_grade_chosen = ""
            return class_grade_chosen
        class_grade_chosen = choose_class_grade()
       
    with col3:
        def choose_class_room():
            class_room = st.selectbox('Phòng', ('A114', 'A115', 'Tất cả'))
            class_shorten = ''
            if class_room == 'A114':
                class_shorten = '114'
            elif class_room == 'A115':
                class_shorten = '115'
            else:
                class_shorten = ''
            return class_shorten
        # st.write(choose_class_room())
        class_shorten = choose_class_room()

    with col4:
        def choose_shift():
            shift = st.selectbox('Buổi', ('Sáng', 'Chiều', 'Tất cả'))
            shift_shorten = ''
            if shift == 'Sáng':
                shift_shorten = 'S'
            elif shift == 'Chiều':
                shift_shorten = 'C'
            else:
                shift_shorten = ''
            return shift_shorten
        shift_shorten = choose_shift()

    with col5:
        st.write('Lớp chuyên')
        check_van = st.checkbox('Văn')
        check_toan = st.checkbox('Toán')
        check_ly = st.checkbox('Lý')
        check_hoa = st.checkbox('Hóa')
        check_anh = st.checkbox('Anh')
        check_tin = st.checkbox('Tin')
        check_sd = st.checkbox('Sử Địa')
        check_trn = st.checkbox('Trung Nhật')
        check_thsn = st.checkbox('TH/SN')
        check_khac = st.checkbox('Khác')
        check_list = [check_van, check_toan, check_ly, check_hoa, check_anh, check_tin, check_sd, check_trn, check_thsn, check_khac]
        check_list_chosen_index = [i for i, x in enumerate(check_list) if x]

        class_label = ['CV', 'CT', 'CL', 'CH', 'CA', 'CTIN', 'CSD', 'CTRN', 'TH|SN', '10A1|10A2|10A3|11A|11B']
        class_tag = ''
        for i in check_list_chosen_index:
            class_tag += str(class_label[i])
            class_tag += '|'
        st.write(class_tag)

    dfc = df[(df['GENDER'].str.contains(gender)) &
            (df['PYTHON-CLASS'].str.contains(class_shorten + "-" + shift_shorten)) &
            (df['CLASS'].str.contains(class_tag, regex=True))
                    ]
    st.write('Số HS:', len(dfc), 
             '(',len(dfc[dfc['GENDER'].str.contains('M')]),'nam,',len(dfc[dfc['GENDER'].str.contains('F')]),'nữ)')
    st.write('GPA:', 'cao nhất:', dfc['GPA'].max(),
             'thấp nhất:', dfc['GPA'].min(),
             'trung bình:', dfc['GPA'].mean().round()
             )
    st.dataframe(dfc)

with tab2:
    subtab1, subtab2 = st.tabs(['Số lượng HS', 'Điểm'])
    with subtab1:
        fig1 = px.pie(df, names = 'PYTHON-CLASS', title = 'Theo lớp AI')
        st.plotly_chart(fig1)
        st.success('Học sinh có xu hướng đăng kí lớp buổi chiều nhiều hơn các lớp buổi sáng. Tuy nhiên sự chênh lệch là không nhiều, nên việc sắp xếp 2 buổi cho HS linh hoạt lựa chọn là hợp lý')

        fig2 = px.pie(df, names = 'REG-MC4AI', title = 'Theo quyết định tiếp tục khóa tiếp theo')
        st.plotly_chart(fig2)
        st.success('Gần 60% HS tiếp tục theo học khóa tiếp theo, là 1 tính hiệu khả quan')
        fig3 = px.pie(df, names = 'GENDER',title='Theo giới tính')
        st.plotly_chart(fig3)
        st.success('Các học sinh nam có hứng thú và đăng kí theo học AI nhiều, gấp 1.5 lần số HS nữ')
    with subtab2:
        session = st.radio('Session', ('S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','GPA'), horizontal = True)
        box = px.box(df, x = 'PYTHON-CLASS', y =session, color='GENDER')
        st.write(box)
        st.success('Nhìn vào GPA tổng kết, ta có một vài kết luận: Các HS Nam tại các lớp có điểm GPA cao hơn các HS nữ. Trong các HS nam, HS nam lớp 114-S có điểm tốt nhất. Trong các HS nữ, HS nữ lớp 115-C có điểm tốt nhất nhưng cũng có nhiều điểm không tốt')

with tab3:
    st.write('x')
    
    def mean_hw(df):
        return df[['S1','S2','S3','S4','S5','S7','S8','S9']].mean()
    df['Homework_avg'] = df.apply(mean_hw, axis = 1)
    # st.write(df['Homework_avg'])

    

    def phan_nhom():
        clean_data()
        
        df['Homework_avg'].fillna(0,inplace = True)
        slider = st.slider('Số nhóm', min_value=2, max_value=5)
        
        X0 = np.array(df['Homework_avg']).reshape(112,1)
        X1 = np.array(df['S6']).reshape(112,1)
        X2 = np.array(df['S10']).reshape(112,1)
        X = np.concatenate((X0,X1,X2), axis = 1)
        

        kmeans = KMeans(n_clusters=slider, n_init='auto')
        kmeans.fit(X)

        kmeans_chart = px.scatter_3d(df,x='Homework_avg', y='S6', z='S10', color=kmeans.labels_)
        st.plotly_chart(kmeans_chart)

        df['LABEL'] = kmeans.labels_

        for i in range(len(np.unique(kmeans.labels_))):
            df_label = df[df['LABEL']==i]
            st.write('Nhóm', i, ':', 'GPA cao nhất:', df_label['GPA'].max(), 'thấp nhất:', df_label['GPA'].min(), 'trung bình:', df_label['GPA'].mean().round())
            st.write(df_label)

    phan_nhom()

with tab4:
    features = st.radio('Số đặc trưng', ('2', '3'), horizontal=True)
    
    def regression():
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        model.fit(X, y)