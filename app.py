import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

def process_video(uploaded_file, model):
    """
    ประมวลผลไฟล์วิดีโอที่อัปโหลดและคืนค่าเส้นทางไปยังไฟล์วิดีโอที่ประมวลผลแล้ว
    """
    try:
        # 1. บันทึกไฟล์ที่อัปโหลดลงใน temp file
        # เราต้องทำสิ่งนี้เพราะ cv2.VideoCapture ต้องการ "เส้นทางไฟล์" จริงๆ
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # 2. เปิดไฟล์วิดีโอด้วย OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("ไม่สามารถเปิดไฟล์วิดีโอได้")
            return None

        # 3. เตรียมไฟล์วิดีโอสำหรับเขียนผลลัพธ์
        # เอาค่า FPS, Width, Height จากวิดีโอต้นฉบับ
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # สร้างไฟล์ชั่วคราวสำหรับผลลัพธ์
        tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = tfile_out.name
        
        # ใช้ 'mp4v' codec สำหรับไฟล์ .mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            st.error("ไม่สามารถสร้างไฟล์ VideoWriter ได้ (Error creating VideoWriter)")
            st.error("ลองตรวจสอบว่า Codec 'avc1' รองรับหรือไม่ หรือลองวิธีที่ 2")
            return None

        # 4. วนลูปอ่านทีละเฟรม
        st.info("กำลังประมวลผลวิดีโอ... กรุณารอสักครู่ ⏳")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 5. ส่งเฟรมเข้าโมเดล YOLO
            # stream=True จะช่วยจัดการหน่วยความจำได้ดีขึ้น
            results = model(frame, stream=True)

            # 6. วาด Bounding Box ลงบนเฟรม
            for r in results:
                # results[0].plot() เป็นฟังก์ชันช่วยของ Ultralytics ที่วาดผลลัพธ์ให้เลย
                annotated_frame = r.plot() 
                
                # 7. เขียนเฟรมที่วาดแล้วลงในไฟล์ใหม่
                out.write(annotated_frame)

        # 8. ปิดไฟล์ทั้งหมด
        cap.release()
        out.release()
        tfile.close()
        tfile_out.close()
        
        # ลบไฟล์ temp ต้นฉบับ
        os.remove(video_path) 

        return output_path

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างประมวลผลวิดีโอ: {e}")
        # พยายามล้างไฟล์ temp หากมีข้อผิดพลาด
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
        return None

# --- ส่วนของ Streamlit UI ---

st.set_page_config(page_title="ระบบตรวจจับหมวกกันน็อค", layout="wide")
st.title("🚧 ระบบตรวจจับหมวกกันน็อค (Helmet Detection)")
st.write("อัปโหลดไฟล์วิดีโอ (.mp4, .mov, .avi) เพื่อเริ่มการตรวจจับ")

# โหลดโมเดล (วางไฟล์ helmet_model.pt ไว้ที่เดียวกับ app.py)
# ใช้ @st.cache_resource เพื่อให้โหลดโมเดลแค่ครั้งเดียว
@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return None

# --- Logic หลักของแอป ---
model_path = "best.pt" # เปลี่ยนชื่อไฟล์ตามโมเดลของคุณ
model = load_yolo_model(model_path)

if model:
    # 1. ช่องสำหรับอัปโหลดไฟล์
    uploaded_file = st.file_uploader(
        "เลือกไฟล์วิดีโอ", 
        type=["mp4", "mov", "avi", "asf", "m4v"]
    )

    if uploaded_file is not None:
        st.video(uploaded_file) # แสดงวิดีโอต้นฉบับ
        
        # 2. ปุ่มเริ่มประมวลผล
        if st.button("เริ่มตรวจจับ (Start Detection)"):
            
            # 3. เรียกใช้ฟังก์ชันประมวลผล
            with st.spinner("กำลังประมวลผล..."):
                processed_video_path = process_video(uploaded_file, model)
            
            # 4. แสดงผลลัพธ์
            if processed_video_path:
                st.success("ประมวลผลเสร็จสิ้น! 🎉")
                
                # อ่านไฟล์วิดีโอที่ประมวลผลแล้วในรูปแบบ bytes
                with open(processed_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                st.video(video_bytes) # แสดงวิดีโอผลลัพธ์
                
                # เสนอให้ดาวน์โหลด
                st.download_button(
                    label="ดาวน์โหลดวิดีโอผลลัพธ์",
                    data=video_bytes,
                    file_name=f"detected_{uploaded_file.name}",
                    mime="video/mp4"
                )
                
                # ลบไฟล์ temp ที่เป็นผลลัพธ์
                os.remove(processed_video_path)
            else:
                st.error("ไม่สามารถประมวลผลวิดีโอได้")
else:
    st.error("ไม่พบไฟล์โมเดล 'helmet_model.pt' กรุณาตรวจสอบว่าวางไฟล์ไว้ถูกต้อง")
