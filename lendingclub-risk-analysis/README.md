# LendingClub Risk Score Analysis

## Project Overview
대출 승인 고객 데이터에서도 일정 비율의 미상환이 발생한다.
기존 승인 기준만으로는 리스크를 완전히 걸러내기 어렵다.

본 프로젝트에서는 LendingClub 데이터를 활용하여
승인 고객 중 미상환 위험이 높은 고객을 식별할 수 있는
리스크 점수를 설계하였다.

## Data
Dataset: LendingClub accepted_2007_to_2018Q4

Target Variable
default_rate = (loan_status != "Fully Paid")

## Analysis Process

1 Data preprocessing
2 Feature selection
3 Feature binning
4 Risk score generation
5 Risk score validation

## Risk Score Design

각 변수의 구간별 평균 부실률을 계산하여
리스크 점수로 활용하였다.

Example variables

- dti
- revol_util
- loan_to_income
- cr_hist_months

## Result

리스크 점수가 높은 그룹일수록
미상환 비율이 유의하게 증가하였다.

이를 통해 승인 고객 내에서도
고위험군을 효과적으로 식별할 수 있었다.

## Business Insight

고위험 구간 고객에 대해

- 추가 심사
- 대출 한도 조정
- 리스크 기반 금리 정책

등의 전략을 적용할 수 있다.
