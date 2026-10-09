# 출처와 공개 범위

김태우의 2025년 머신러닝 수업 프로젝트 「나만의 음성인식 모델 만들기」입니다. `ML_Project.ipynb`와 최종 발표자료 4쪽을 참고했습니다. 학번과 발표자료 원본은 공개하지 않습니다.

집계 수치는 노트북 저장 출력에서 옮겼으며 그래프는 8 epoch 로그로 다시 그렸습니다. 재학습 결과가 아닙니다. 발표와의 차이는 결과 문서에 구분했습니다.

- [OpenAI Whisper](https://github.com/openai/whisper): 사전학습 인코더와 오디오 로드·Mel 필터. 기반 코드와 모델은 Whisper의 [MIT 라이선스](https://github.com/openai/whisper/blob/main/LICENSE)를 따릅니다.
- [Whisper 오디오 구현](https://github.com/openai/whisper/blob/main/whisper/audio.py): 공식 전처리 경로와 비교할 때 참고했습니다.
- PyTorch: 학습·모델 구현·과거 Lite 내보내기
- scikit-learn: 분류 보고서·혼동행렬

프로젝트 자체의 코드·문서에는 별도의 오픈소스 라이선스를 지정하지 않았습니다. 의존성의 라이선스가 프로젝트 전체에 자동 적용되는 것은 아닙니다. 녹음 자료의 이용 권한은 별도 확인해야 합니다.
