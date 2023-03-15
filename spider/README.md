 🕸️🕷️ Spider

ก่อนจะเริ่มใช้งาน scrapy ต้องเลือก directory ที่จัดเก็บไฟล์ที่จะได้จากการ scape ไว้ 
ขั้นตอนแรกของการทำงาน spider คือกำหนดคลาสที่จะเป็นขอบเขตในการขูดข้อมูลจากเว็บไซต์ manootchecklist 
parse() เป็น Method ที่จะทำการขูดข้อมูลจากเว็บไซต์ที่กำหนด ตามรูปแบบที่ได้สร้างไว้ และนำไปเก็บไว้ในตัวแปรที่สร้างขึ้นไว้เพื่อเก็บข้อมูล จากนั้นขูดข้อมูลสำเร็จจะทำการเปลี่ยนหน้าเว็บไซต์ไปเรื่อยๆ จนกว่าจะไม่เจอเว็บไซต์ในหน้าถัดไป จึงจะหยุดการทำงานของ spider

หากต้องการให้ spider ใช้งาน scrapy ทำได้โดย :

```
pip install scrapy
```

คำสั่งนี้เป็นการใช้งาน libary ของ scrapy 

หากต้องการให้ spider ทำงาน ให้ไปที่ directory ระดับบนสุดของโปรเจ็กต์แล้วเรียกใช้ : 

```
scrapy runspider manootchecklist-spider.py -o manoonchecklist-lyrics.json
```

คำสั่งนี้เรียกใช้ spider ด้วยชื่อไฟล์ `manootchecklist-spider.py` ที่เพิ่งเพิ่ม ซึ่งจะส่งคำขอการขูดไปที่เว็บไซต์ [manootchecklist.wordpress.com](https://manootchecklist.wordpress.com) ระหว่างที่สคริปต์นี้จะสร้างไฟล์ json file ชื่อ `manoonchecklist-lyrics.json` ไฟล์๋นี้จะประกอบไปด้วยข้อมูลจากเว็บไซต์ [manootchecklist.wordpress.com](https://manootchecklist.wordpress.com) โดยที่ข้อมูลจะเพิ่มขึ้นเรื่อย ๆ ระหว่างการทำงาน

เมื่อได้ข้อมูลแล้วนั้น ให้ทำการสั่งคำสั่งต่อไปนี้

```
python spider/isolate.py -s source_input.json -d destination_output.csv
```

คำสั่งนี้จะสร้างตารางขึ้นมา โดยจะวางภาษาไทยและภาษาอังกฤษคู่ขนานกัน