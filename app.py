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
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("ไม่สามารถเปิดไฟล์วิดีโอได้")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # --- 💡 แก้ไขจุดที่ 1: เปลี่ยน Suffix เป็น .webm ---
        tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
        output_path = tfile_out.name
        
        # --- 💡 แก้ไขจุดที่ 2: เปลี่ยน Codec เป็น VP90 (สำหรับ WebM) ---
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # เพิ่ม Fallback หาก VP90 ใช้งานไม่ได้ (ลอง VP80)
        if not out.isOpened():
            st.warning("Codec 'VP90' ไม่ทำงาน, กำลังลอง 'VP80'...")
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                st.error("ไม่สามารถสร้างไฟล์ VideoWriter ได้ (ทั้ง VP90 และ VP80 ล้มเหลว)")
                return None

        st.info("กำลังประมวลผลวิดีโอ... กรุณารอสักครู่ ⏳")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, stream=True)
            for r in results:
                annotated_frame = r.plot() 
                out.write(annotated_frame)

        cap.release()
        out.release()
        tfile.close()
        tfile_out.close()
        
        os.remove(video_path) 
        return output_path

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างประมวลผลวิดีโอ: {e}")
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
        return None

# --- ส่วนของ Streamlit UI ---

st.set_page_config(page_title="ระบบตรวจจับหมวกกันน็อค", layout="wide")
st.title("🚧 ระบบตรวจจับหมวกกันน็อค (Helmet Detection)")
st.write("อัปโหลดไฟล์วิดีโอ (.mp4, .mov, .avi) เพื่อเริ่มการตรวจจับ")

@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return None

model_path = "best.pt" 
model = load_yolo_model(model_path)

if model:
    uploaded_file = st.file_uploader(
        "เลือกไฟล์วิดีโอ", 
        type=["mp4", "mov", "avi", "asf", "m4v"]
    )

    if uploaded_file is not None:
        st.video(uploaded_file) 
        
        if st.button("เริ่มตรวจจับ (Start Detection)"):
            
            with st.spinner("กำลังประมวลผล..."):
                processed_video_path = process_video(uploaded_file, model)
            
            if processed_video_path:
                st.success("ประมวลผลเสร็จสิ้น! 🎉")
                
                try:
                    with open(processed_video_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    
                    # --- 💡 แก้ไขจุดที่ 3: ระบุ format เป็น 'video/webm' ---
                    st.video(video_bytes, format='video/webm') 
                    
                    st.download_button(
                        label="ดาวน์โหลดวิดีโอผลลัพธ์",
                        data=video_bytes,
                        file_name=f"detected_{uploaded_file.name}.webm", # เปลี่ยนชื่อไฟล์
                        mime="video/webm" # เปลี่ยน Mime type
                    )
                    
                    os.remove(processed_video_path)

                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการอ่านหรือแสดงผลไฟล์วิดีโอ: {e}")
                    if os.path.exists(processed_video_path):
                        os.remove(processed_video_path)
            else:
                st.error("ไม่สามารถประมวลผลวิดีโอได้")
else:
    st.error("ไม่พบไฟล์โมเดล 'helmet_model.pt' กรุณาตรวจสอบว่าวางไฟล์ไว้ถูกต้อง")
