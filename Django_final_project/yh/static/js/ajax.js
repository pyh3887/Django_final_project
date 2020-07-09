
$(document).ready(function(){
	
	$('#Progress_Loading').hide(); //첫 시작시 로딩바를 숨겨준다.
	$('#Progress_Loading2').hide();
	
	$("#yh_anal1").bind('click',function(){
		 var raisehandcount = $("#raisehandcount").val()
		var enterclass = $("#enterclass").val()
	      var checkcount = $("#checkcount").val()
	      var discount = $("#discount").val()
		
		
		 //ajax실행시 로딩바를 보여준다.
				$("#showData2").empty()
				$("#showData1").empty()
				$('#Progress_Loading').show();
		
		//alert('b')
			$.ajax({
				
				url:'show',
				type:'get',
				dataType:'json',
				async:true,
				data:{'raisehandcount':raisehandcount,'checkcount':checkcount,'discount':discount,'enterclass':enterclass},
				success:function(data){
					//alert(data)
					//alert(data['a'])
					str1 = "<h1 style='font-family': 'Noto Sans KR', sans-serif'>"+ data['a'] + "</h1>"
					str1 += "<h1 style='font-family': 'Noto Sans KR', sans-serif'>정확도 :"+ data['accuracy'] + "%</h1>"
					str2 = "<div style='font-family': 'Noto Sans KR', sans-serif'>" + data['yh_grap3']+ "</div>" 					
					$('#showData1').append(str1)	
					$('#showData2').append(str2)	
					$('#Progress_Loading').hide(); //ajax종료시 로딩바를 숨겨준다.		
				},
				error:function(){
					$('#showData').text('error')
				}
			})
		})
	
	$("#btnSubmit").bind('click',function(){
	      
	      // alert("버튼 넘어옴")
	      var raisehandcount = $("#raisehandcount").val()
	      var checkcount = $("#checkcount").val()
	      var discount = $("#discount").val()
	     alert(raisehandcount + checkcount + discount)
	      $("#predictShow").empty()
	      $.ajax({
	         url:'predictgo',
	         type:'get',
	         data:{'raisehandcount':raisehandcount,'checkcount':checkcount,'discount':discount},
	         dataType:'json',
	         success:function(data){
	            alert('result' + data['resultPredict'])
	            data['figs']
	            
	            $("#predictShow").html(data['fig1']).css("color", "red");
	         },
	         error:function(){
	            alert('실패')
	         }
	      })   
	   })
	$.ajax({
	      url:'show3',
	      type:'get',
	      dataType:'json',
	      async:true,
	      success:function(data){
	         //alert(data)
	         $('.wait').detach()
	         maegraph = data['ks_graph_mae']
//	         mae = "<h4>" + data['mae'] + "</h4><br><br>"
	         msegraph = data['ks_graph_mse']
	         mse = "<h4>" + data['mse'] + "</h4><br><br>"
	         lineargraph = data['ks_graph_linear']
	         errgraph = data['ks_graph_err']
	         logitgraph = data['ks_graph_logit']
	         linear = "<h1>Linear Regression</h1>"
	         logit = "<h1>Logistic Regression</h1>"
	         
//	            모델을 최적화하기 위해서는 모델의 성능을 평가할 수 있는 척도가 필요하며, 모델이 예측한 값과 관측값 사이의 불일치 정도를 확인하려면 '거리'를 반영하는 척도가 필요하다. 이러한 거리를 loss(손실) 함수라고 하며, 이 함수의 값을 최소화하는 파라미터(가중치, 편향)을 찾아내는 것이 딥러닝 모델을 최적화하는 것이다.<br>\
	         maetext = "<h4>발표수와 상관 계수가 0.3보다 큰 것들을 선택해서 상관 관계가 높은 칼럼들을 선택해 linear regression을 사용해 학습을 하여 모델을 만들어서 분류 한다.<br>\
	                      <br>" + data['loss'] + "<br>" + data['mae'] + "<br><br>\
	                       MAE - mean absolute error -   평균절대오차<br>\
	                                              평균 절대 오차(MAE) 는 모든 절대 오차의 평균이다.<br>\
	                       MAE는 회귀지표로써 사용된다.</h4>"
	                      
	           msetext = "<h4>MSE- mean squared error - 평균 제곱 오차 <br>\
	                                            잔차(오차)의 제곱에 대한 평균을 취한 값<br>\
	                                            통계적 추정의 정확성에 대한 질적인 척도<br>\
	                                            수치가 작은수록 정확성이 높은 것<br>\
	                      MSE(평균제곱오차)가 추정정확도의 척도로 많이 사용되는 이유 <br>\
	                      1. 수학적인 분석이 쉽다 2. 계산의 용이하다 등<br>\
	                      MSE는 손실함수로써 사용된다.</h4>"
	           
	         x = "<h4>예측 값 : " + data['x'] + "</h4><br>"
	         y = "<h4>실제 값 : " + data['y'] + "</h4><br>"
	         lineartext = "<br><h4>예측 값과 실제 값으로 다음과 같은 선형 회귀선 그래프로 시각화 하여 확인 할 수 있다.</h4>"
	            
	         errtext = "<h4>잔차항의 정규분포성을 이루는지 확인 할 수 있다.<br>\
	                                            그래프를 보면 자료가 부족한 상태에서 강제로 자료를 복사 붙여 넣기 하면서 늘렸기 때문에 정규성을 만족하기에는 조금 부족해 보이는 모양의 그래프가 출력 된다.</h4><br>"
	         
	         logittext1 = "<h4>담당부모의 상관 계수가 0.3보다 큰 것들을 선택해서 상관 관계가 높은 칼럼들로 logistic regression을 사용해 학습을 하여 모델을 만들어서 분류 한다.<br>\
	                                        담당부모는 아빠를 0, 엄마를 1로 주고 예측해야 하는 값이 두 개 밖에 안되므로 sigmoid를 사용해 이항 분류를 한다.</h4><br>"
	         acc = "<h4>정확도 : " + data['acc'] + "</h4><br>"
	         xh = "<h4>예측 값 : " + data['xh'] + "</h4><br>"
	         yh = "<h4>실제 값 : " + data['yh'] + "</h4><br>"
	         logittext2 = "<h4>옆의 그래프는 학습시키는 과정에서 epoch이 증가함에 따라 loss(손실)가 줄어드는 것을 시각화 한 것이다.</h4><br>"
	         
	         
	         $('#maegraph').append(maegraph)
//	         $('#maetext').append(mae)
	         $('#maetext').append(maetext)
	         
	         $('#msegraph').append(msegraph)
	         $('#msetext').append(mse)
	         $('#msetext').append(msetext)
	         
	         
	         $('#lineargraph').append(lineargraph)
	         $('#lineartext').append(y)
	         $('#lineartext').append(x)
	         $('#lineartext').append(lineartext)
	         
	         $('#errgraph').append(errgraph)
	         $('#errtext').append(errtext)
	         
	         $('#logitgraph').append(logitgraph)
	         $('#logittext').append(logittext1)
	         $('#logittext').append(acc)
	         $('#logittext').append(yh)
	         $('#logittext').append(xh)
	         $('#logittext').append(logittext2)
	            
	         $('.kstitle').append(linear)
	         $('.kstitle2').append(logit)
	      },
	      error:function(){
	         $('#showData').text('error')
	      }
	   })
	//alert("a")
	//버거를 눌렀을 시
	
		
	$("#anal2").bind('click',function(){
		$('#Progress_Loading2').show(); //ajax실행시 로딩바를 보여준다.
		$("#showData").empty()
		//alert('b')
			$.ajax({
				url:'show2', 
				type:'get',
				dataType:'json',
				async:true,
				success:function(data){
					//alert(data)
					alert(data['anal'])
					str1 = "<h1>"+ data['anal'] + "</h1>"					
					$('#showData4').append(str1)
					$('#Progress_Loading2').hide(); //ajax종료시 로딩바를 숨겨준다.					
				
				},
				error:function(){
					$('#showData').text('error')
				}
			})
		})
		
		

})
