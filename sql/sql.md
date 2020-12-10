# 하루 10분 SQL 

## 08장 기본 명령어

### 8.1 SELECT

구문 #1 SELECT * FROM 테이블명;

구문 #2 필드명1, 필드명2 ..., 필드명n FROM 테이블명;

구문 #3 vlfemaud [AS] 별명 FROM 테이블명;



예제 #1 고객 테이블의 전체 내용을 검색한다.


- SQL
SELECT
    *
FROM tb_customer;

- 결과
![](https://user-images.githubusercontent.com/72365762/97836873-915ecc00-1d20-11eb-87e3-74a5ec386d26.png)

- 설명
SELECT 명령에 '*' 문자를 지정해 고객 테이블의 전체 내용을 검색했다. 이SQL은 테이블의 모든 내용을 검색하는 가장 기본적인 명령이다.

예제 #2 고객 테이블에서 고객코드, 고객명, 전화번호, 이메일을 검색한다.

- SQL
SELECT customer_cd,
       customer_nm,
       phone_number,
       email
FROM tb_customer;    

- 결과
![](https://user-images.githubusercontent.com/72365762/97837091-ed295500-1d20-11eb-9b97-628c45d15b06.png)

- 설명
고객 테이블(tb_customer)에서 고객코드(customer_cd), 고객명(customer_nm), 전화번호(phone_number), 이메일(email)을 검색했다. 이 명령은 필드가 많은 테이블에서 필요한 내용만 검색한다.

예제 #3 고객 테이블에서 고객코드, 고객명, 전화번호, 이메일을 검색하면서 필드 제목을 한글로 바꾼다.

- SQL
SELECT customer_cd,
       customer_nm,
       phone_number,
       email
FROM tb_customer;    

- 결과
![](https://user-images.githubusercontent.com/72365762/97837123-f9151700-1d20-11eb-9fbd-723eac715f7c.png)

- 설명
고객 테이블(tb_customer)에서 고객코드(customer_cd), 고객명(customer_nm), 전화번호(phone_number), 이메일(email)을 검색하면서 영문 필드명인 customer_cd는 '고객코드'등으로 바꿨다.

### 8.2 WHERE

구문 #1 SELECT 검색필드명

구문 #2 FROM 테이블명

구문 #3 WHERE 조건식;

예제 #1 고객 테이블에서 남성인 고객을 검색한다.

- SQL
SELECT
    *
FROM tb_customer
WHERE mw_flg = 'M';    

- 결과
![](https://user-images.githubusercontent.com/72365762/97837164-05996f80-1d21-11eb-984e-9a26fd3875e8.png)


- 설명
고객 테이블(tb_customer)에서 성별(MW_flg)이 남성('M')인 조건으로 검색했다. 성별이 모두 'M'인지 확인한다. 성별은 'M'이 남성이고 'W'는 여성이다.

예제 #2 고객 테이블에서 고객명이 '김한길'인 고객을 검색한다.

- SQL
SELECT
    *
FROM tb_customer
WHERE customer_nm = '김한길';   

- 결과
![](https://user-images.githubusercontent.com/72365762/97837184-11853180-1d21-11eb-9eec-963f31246355.png)

- 설명
고객 테이블(tb_customer)에서 고객명(customer_nm)이 '김한길'인 조건으로 검색했다.

### 8.3 AND

구문 #1 SELECT 검색필드명

구문 #2 FROM 테이블명

구문 #3 WHERE 조건식1 AND 조건식2 AND ... 조건식n;

예제 #1 고객 테이블에서 2019년 이후 등록한 여성 고객을 검색한다.

- SQL
SELECT
    *
FROM tb_customer
WHERE customer_cd > '2019000'
and mw_flg = 'W';   

- 결과
![](https://user-images.githubusercontent.com/72365762/97837221-1fd34d80-1d21-11eb-9d4d-503171509304.png)


- 설명
고객 테이블(tb_customer)에서 고객코드(customer_cd)가 '2019000'보다 크고 성별(MW_flg)이 여성('W')인 조건으로 검색했다. 

예제 #2 고객 테이블에서 남성이면서 출생 연도가 '1990'년 미만인 고객을 검색한다.

- SQL
SELECT
    *
FROM tb_customer
WHERE birth_day < '19900101'
and mw_flg = 'M';   

- 결과
![](https://user-images.githubusercontent.com/72365762/97837257-2c57a600-1d21-11eb-9b53-779ff00b44b4.png)

- 설명
고객 테이블(tb_customer)에서 성별(MW_flg)이 남성('M')이고 생년월일(birth_day)이 '19900101' 미만인 조건으로 검색했다.

### 8.4 OR

구문 #1 SELECT 검색필드명

구문 #2 FROM 테이블명

구문 #3 WHERE 조건식1 OR 조건식2 OR ... 조건식n;

예제 #1 고객 테이블에서 생년월일이 '19900101' 이후거나 누적포인트가 20,000이상인 고객을 검색한다.

- SQL
SELECT
    *
FROM tb_customer
WHERE birth_day >= '19900101'
OR total_point >= 20000;   

- 결과
![](https://user-images.githubusercontent.com/72365762/97837280-38dbfe80-1d21-11eb-8b3e-b69d956da02d.png)


- 설명
고객 테이블(tb_customer)에서 생년월일(birth_day)이 '19900101' 이후거나, 누적포인트(total_point)가 20,000 이상인 조건으로 검색했다. 

예제 #2 고객 테이블에서 남성인 고객 중 생년월일이 '19700101' 이전이거나 누적포인트가 20,000 이상인 고객을 검색한다.

- SQL
SELECT
    *
FROM tb_customer
WHERE mw_flg = 'M'
and (birth_day < '19700101'
OR total_point >= 20000);  

- 결과
![](https://user-images.githubusercontent.com/72365762/97837294-44c7c080-1d21-11eb-8e60-45b80c723067.png)

- 설명
고객 테이블(tb_customer)에서 성별(MW_flg)이 남성('M')인 고객 중  생년월일(birth_day)이 '19700101' 이전이거나 누적포인트(total_point)가 20,000 이상인 조건으로 검색했다.

### 8.5 BETWEEN .. AND

구문 #1 SELECT 검색필드명

구문 #2 FROM 테이블명

구문 #3 WHERE 필드명 [NOT] BETWEEN 시작값 AND 종료값;

예제 #1 고객 테이블에서 여성이고 누적포인트가 10,000에서 20,000 이하인 고객을 검색한다.

- SQL
SELECT
    *
FROM tb_customer
WHERE mw_flg = 'W'
and total_point BETWEEN 10000 AND 20000;  

- 결과
![](https://user-images.githubusercontent.com/72365762/97837647-f7981e80-1d21-11eb-855a-11478551667a.png)

- 설명
고객 테이블(tb_customer)에서 성별(MW_flg)이 여성('W')이고 누적포인트(total_point)가 10,000과 20,000 사이인 고객을 검색했다. 

예제 #2 고객 테이블에서 생년월일이 '19800101'과 '19891231' 사이인 고객을 검색한다.

- SQL
SELECT
    *
FROM tb_customer
WHERE birth_day BETWEEN '19800101' AND '19891231'; 

- 결과
![](https://user-images.githubusercontent.com/72365762/97838128-e8fe3700-1d22-11eb-8c2e-d1813d6c266d.png)

- 설명
고객 테이블(tb_customer)에서 생년월일(birth_day)이 '19800101'과 '19891231' 사이에 속하는 고객을 검색했다.