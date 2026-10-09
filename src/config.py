"""원본 노트북의 명령어 순서와 오디오 설정."""
LABEL_MAP = {
    '1-1': '불 켜', '1-2': '불 꺼', '2-1': '시작 해', '2-2': '정지 해',
    '3-1': '앞으로 가', '3-2': '뒤로 가', '4-1': '음악 틀어',
    '5-1': '전화 받아', '6-1': '모델 학습', '7-1': '데이터 준비', '8-1': '필터 적용',
}
ORDERED_LABELS = list(LABEL_MAP.values())
LABEL_TO_ID = {name: i for i, name in enumerate(ORDERED_LABELS)}
ID_TO_LABEL = dict(enumerate(ORDERED_LABELS))
NUM_LABELS = len(ORDERED_LABELS)
TARGET_SR = 16000
TARGET_SECONDS = 3.0
TARGET_SAMPLES = 48000
