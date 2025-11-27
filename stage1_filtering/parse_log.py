from Drain import LogParser

input_dir = '/home/bigdatanai/hhhhaha/data1111'  # The input directory of log file
output_dir = '/home/bigdatanai/hhhhaha/data1111/result'  # The output directory of parsing results
log_file = '/home/bigdatanai/hhhhaha/data1111/243456582_logcatAPI 30 이후의 Android 릴리스가 M1 Mac의 에뮬레이터에서 시작되지 않습니다..txt'  # The input log file name
dataset = 'Android'  # The name of the dataset
hdfs_format = r'<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
bgl_format = r'<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
openstack_format = r'<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
thuderbird_format = r"<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>"
android_format= r'<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>'
bgl_regex = [
    r'core\.\d+',
    r'(?:\/[\*\w\.-]+)+',  # path
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'0x[0-9a-f]+(?: [0-9a-f]{8})*',  # hex
    r'[0-9a-f]{8}(?: [0-9a-f]{8})*',
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
]

hdfs_regex = [
    r'blk_(|-)[0-9]+',  # block id
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
]

openstack_regex = [
    r'(?<=\[instance: ).*?(?=\])',
    r'(?<=\[req).*(?= -)',
    r'(?<=image ).*(?= at)',
    r'(?<=[^a-zA-Z0-9])(?:\/[\*\w\.-]+)+',  # path
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'(?<=\s|=)\d+(?:\.\d+)?'
]

Android_regex = [
            r"(/[\w-]+)+",
            r"([\w-]+\.){2,}[\w-]+",
            r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",
]
thunderbird_regex = [r"(\d+\.){3}\d+"]
threshold = 5
delimeter  = [r""]

# 모듈로 import될 때는 실행하지 않도록 조건부 처리
if __name__ == "__main__":
    parser = LogParser(logname=dataset, log_format=android_format, indir=input_dir, 
                       outdir=output_dir, threshold=threshold, delimeter=delimeter, rex=Android_regex)
    parser.parse(log_file)