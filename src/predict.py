import hydra
import torch
from transformers import AutoTokenizer

from dataloader import get_dataloader, load
from models.MainModel_PL import PLEncoder


@hydra.main(config_name="config.yml")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.PATH.tokenizer)

    sentence = "중소벤처기업부가 인공지능 빅데이터 스마트헬스케어 스마트공장 등 4차산업혁명 관련 유망 예비창업자 600명을 선발 최대 1억 원을 지원한다  자금 지원 뿐 아니라 전담멘토를 매칭 경영  자문 서비스를 제공하고 창업교육도 40시간 시켜준다 16일 중기부는 2020년 예비창업패키지 특화분야에 참여할 예비창업자를 이달 17일부터 다음달 20일까지 모집한다고 밝혔다 예비창업패키지는 혁신적 기술창업 아이디어를 가진 예비창업자의 원활한 초기 창업사업화를 위해 사업화 자금과 창업교육 멘토링을 패키지로 지원하는 사업으로 이번에 특화 분야를 모집한다  모집 인원은 600명 내외다 앞서 일반분야는 지난 3월 16일 1100명 선발을 위한 신청  접수를 완료했다 특화 분야는 인공지능 빅데이터 스마트 헬스케어 스마트공장 블록체인 스마트시티 자율주행 등 4차산업혁명과 관련한 첨단 기술 분야다  지능형반도체 5G 드론 분야는 올해 처음 생겼다  특히 선발은 4차산업혁명 관련 10개 부처가 추천한 16개 주관기관에 일임 이들 기관이 선발한다  인공지능 경우 광주과학기술원이 40명을 선발한다 신청자격은 일반분야와 동일하게 공고일 기준 창업 경험이 없거나 신청자 명의의 사업체를 보유하고 있지 않은 사람이다  폐업 경험이 있는 경우 이종 업종 제품이나 서비스를 생산하는 기업으로 창업해야 한다  거주지 창업예정지 등에 관계없이 자신의 창업아이템 분야에 맞는 주관기관을 1개 선택해 신청하면 된다 선정평가는 창업아이템 개발 동기 사업화 전략 시장분석  경쟁력 확보방안 대표자와 팀원의 보유역량 등을 서류  발표로 한다  주관기관별 지원규모에 따라 발표평가 고득점자 순으로 최종 선정자를 결정한다  단 코로나19 확산 추이에 따라 발표평가는 온라인 평가로 대체될 수 있다"

    data = tokenizer(
        sentence,
        max_length=cfg.DATASETS.seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    print(f"data: {data}")

    # # model
    # model = torch.load(cfg.PATH.model)
    # model.eval()

    # output = model(e)
    # print(f"output : {output}")


if __name__ == "__main__":
    main()
