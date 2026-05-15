# LIBRA Benchmark Scenarios

이 폴더는 중간발표용 controlled scenario benchmark 입력이다.

- `profiles.yaml`: 공통 사용자 프로필과 제약 조건.
- `stock_universe.yaml`: KRX 종목 풀, 섹터, ESG 점수.
- `scenarios/*.yaml`: 10개 검증 시나리오.

`expected`는 정답이 아니라 실행 전 가설이다. 실제 Judge 호출 결과가 다르면 시나리오를 결과에 맞추지 말고, 왜 다른 판단이 나왔는지 분석한다.

실행:

```powershell
D:\Libra\.venv\Scripts\python.exe scripts\run_benchmark.py --skip-libra
D:\Libra\.venv\Scripts\python.exe scripts\run_benchmark.py --base-url http://127.0.0.1:8010
```

결과는 `outputs/benchmark/<timestamp>/` 아래에 저장된다.
