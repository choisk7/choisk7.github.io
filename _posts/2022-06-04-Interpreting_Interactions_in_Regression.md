---
layout: single
title:  "Interpreting Interactions in Regression"
categories: ML
tag: [Polynomial, interaction terms]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: false
---
**[공지사항]** ["출처: https://www.theanalysisfactor.com/interpreting-interactions-in-regression/"](https://www.theanalysisfactor.com/interpreting-interactions-in-regression/)
{: .notice--danger}



# Interpreting Interactions in Regression
regression model에 interaction terms을 추가하면 실질적인 이점이 있습니다. 모델의 변수 간의 관계에 대한 이해를 크게 확장시켜 줍니다. 그리고 더 구체적인 가설을 테스트할 수 있습니다. 그러나 regression에서 상호작용을 해석하려면 각 계수가 말하는 내용을 이해해야 합니다.

회귀 계수 해석의 예는 토양에 있는 박테리아의 수와 관목이 부분적으로 또는 전체적으로 햇빛을 쬐는지 여부로 관목의 높이를 나타내는 모델이 있습니다. 높이는 cm로 측정되고, 박테리아는 토양의 ml당 천 단위로 측정되며, 식물이 부분으로 햇빛을 쬐는 경우 Sun = 0, 완전히 햇빛을 쬐는 경우 Sun = 1입니다.

회귀 방정식은 다음과 같이 추정되었습니다:

<p align="center"><img src="/assets/images/220605/1.png"></p>


$$Height = 42 + 2.3*Bacteria + 11*Sun$$

## How adding an interaction changes the model

관목 높이에 대한 토양에서의 박테리아의 수 사이의 관계가 햇빛을 전체적으로 쬐는 경우보다 부분적으로 햇빛을 쬐는 경우가 다르다는 것을 가설로 검정하려는 경우 모델에 interaction terms을 추가하는 것이 유용할 것입니다.

한 가지 가능성은 전체적으로 햇빛을 쬐는에서 토양에 더 많은 박테리아가 있는 관목의 높이가 더 커지는 경향이 있다는 것입니다. 그러나 부분적으로 햇빛을 쬐는 경우에서는 토양에 더 많은 박테리아가 있는 식물이 더 높이가 낮습니다.

또 다른 가능성은 토양에 더 많은 박테리아가 있는 관목은 햇빛 여부 상관없이 모두에서 키가 더 큰 경향이 있다는 것입니다. 그러나 이러한 관계는 전체적으로 햇빛을 쬐는 경우가 훨씬 더 극적입니다.

상호작용이 있다는 것은 반응 변수에 대한 한 예측 변수의 효과가 다른 예측 변수의 다른 값에서 다르다는 것을 나타냅니다. 두 예측 변수를 곱해 모델에 interaction terms을 추가해 검정을 해봅니다. 회귀 방정식은 다음과 같습니다:

<p align="center"><img src="/assets/images/220605/2.png"></p>

$$Height = B0 + B1*Bacteria + B2*Sun + B3*Bacteria*Sun$$

모델에 interaction terms을 추가하면 모든 계수의 해석이 크게 바뀝니다. interaction terms가 없으면 B1을 높이에 대한 박테리아의 고유한 효과로 해석합니다.

그러나 상호 작용은 높이에 대한 박테리아의 영향이 햇빛의 여부에 대해 다르다는 것을 의미합니다. 따라서 높이에 대한 박테리아의 고유한 효과는 B1에 국한되지 않습니다. B3 및 Sun의 값에 따라 달라집니다.

Bacteria의 영향은 B1 + B3\*Sun에서 Bacteria를 곱한 것으로 표현됩니다. B1은 이제 Sun = 0일 때만 높이에 대한 박테리아의 고유한 영향으로 해석됩니다.

## Interpreting the Interaction

여기의 예시에서 interaction terms을 추가하면 모델은 다음과 같습니다.

$$Height = 35 + 4.2*Bacteria + 9*Sun + 3.2*Bacteria*Sun$$

interaction terms을 추가해 B1과 B2의 값이 변경되었습니다. 높이에 대한 박테리아의 영향은 이제 4.2 + 3.\*Sun입니다. 부분적으로 햇빛을 쬐는 관목의 경우 Sun = 0이므로 박테리아의 효과는 4.2 + 3.2\*0 = 4.2입니다. 따라서 부분적으로 햇빛을 받는 두 식물의 경우 토양에 1000 bacteria/ml가 더 많은 관목은 박테리아가 적은 관목보다 4.2cm 더 클 것으로 예상됩니다.

그러나 햇볕을 완전히 쬐는 경우 박테리아의 영향은 4.2 + 3.2\*1 = 7.4입니다. 따라서 햇볕이 잘 드는 두 관목의 경우 토양에 1000 bacteria/ml가 더 많은 식물은 박테리아가 적은 식물보다 7.4cm 더 클 것으로 예상됩니다.

상호 작용으로 인해 식물이 햇빛을 쬐는 여부에 따라 토양에 있는 박테리아의 효과가 달라집니다. 이것은 또, 높이와 박테리아 수 사이의 기울기가 Sun의 값에 따라 다르다는 것입니다. B3는 이러한 기울기가 얼마나 다른지를 나타냅니다.

B2를 해석하는 것은 더 어렵습니다. B2는 Bacteria = 0일 때의 Sun의 영향입니다. Bacteria는 연속형 변수이므로 0인 경우가 거의 없습니다. 따라서 B2는 그 자체로는 사실 의미가 없다는 것을 의미할 수 있습니다.

박테리아의 영향 대신에, Sun의 영향을 이해하는 것이 더 유용할 수 있지만 위에서 언급했다시피 어려울 수 있습니다. Sun의 영향은 B2 + B3\*Bacteria로, Bacteria의 값마다 달라집니다. 이러한 이유로 종종 Sun의 영향을 직관적으로 이해하는 유일한 방법은 몇 가지 박테리아 값을 방정식에 넣어 종속 변수인 높이가 어떻게 변하는지 확인하는 것입니다.