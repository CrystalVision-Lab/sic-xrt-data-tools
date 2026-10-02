# AI 시각 타입 제안 v1

prepare는 review_priority_plan v1의 작업 해시·ID 목록을 확인하고 별도 출력에 원본 주변
256픽셀 패치와 64픽셀 영역의 최근접 2배 확대 패널을 만듭니다. 기존 라벨은 패널에서 숨기고,
좌표 표시를 비운 중심과 표시 없는 확대 영상을 함께 봅니다. 가장자리에서는 원본 좌표를
이동하지 않으며 영상 바깥 공간은 문맥으로 해석하지 않습니다. 원본 값은 변경하지 않습니다.

각 blind_index를 실제로 관찰한 뒤 status=suggest/hold, proposed_base_label=BPD/TED/TSD/null,
strength=low/moderate, reason을 제출합니다. strength는 정확도나 보정된 확률이 아닙니다.
같은 모델의 기존 예측을 재확인으로 사용하지 않으며 전체 207건을 실제로 보지 않고
복사하거나 제공자 라벨을 그대로 승인하지 않습니다.

finalize는 모든 패널에 정확히 한 판정이 있는지와 보류 타입 없음·근거 필수·사람/전문가
확정과 세부 문자 금지를 검사합니다. 기존 제공자 라벨과 비교는 판독 후에 합니다.
ai_visual_type_proposals v1 JSON/CSV/HTML과 해시를 저장합니다. 실제 판독은 Codex_AI,
표시 이름은 요청한 양희승입니다. human_verified/expert_semantic_confirmation/
eligible_for_verified_evaluation/training_ready는 false입니다. 타입은 형태상 기본 종류
제안이며 회절 조건·다중 관측에 의한 물리적 동정은 아닙니다. TSD와 혼합 전위 구별도
이 출력에서 확정하지 않습니다. 좌표가 다른 결함을 가리키거나 영상 품질이 불충분하면
타입을 붙이지 않고 보류합니다. 세부 a~f/a~c는 정의와 대응 자료 없이 추정하지 않습니다.

판정 입력은 blind_index/status/proposed_base_label/strength/reason 다섯 필드만 허용하며
좌표·원본 라벨·판독자·평가 자격을 덮어쓸 수 없습니다. HTML은 원본 패치와 정확한
주석 좌표에 표시한 같은 패치를 나란히 보여 줍니다. 중심 여백을 둔 십자가는 이미지
별도 사본에만 그립니다. 보류·종류 제안·원래 기본 타입과 다른 제안으로 필터링할 수
있으며, 기존 라벨을 숨긴 확대 패널도 연결합니다. 출력 해시는 표시 사본까지 포함합니다.

참고 논문: Harada, Matsubara, Murayama, Diamond and Related Materials 138 (2023) 110192,
https://doi.org/10.1016/j.diamond.2023.110192 . 원고 전체 접근이 제한되면 실제 읽은
초록·공개 섹션의 범위를 넘어 논문을 완독했다고 주장하지 않습니다.

원본 라벨·사람 검수 세션·기존 모델·시험셋은 변경하지 않고 학습을 실행하지 않습니다.
전체 영상·미주석 웨이퍼·3D·전후 대응을 판독 완료한 것으로 확대하지 않습니다.
