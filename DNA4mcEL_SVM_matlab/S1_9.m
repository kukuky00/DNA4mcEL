function [M]=S1_9()
M=zeros(3108,59*41);
A=textread('S1.txt','%s');
B=char(A);
for i=1:3108
    E(i,:)=B(i*2,:);
end
b=size(E,1);      %核小体的样本的个数（即列数）
a=41;

for i=1:b
    for j=1:a-1
        m=0;
        if E(i,j)=='A'
            M(i,(j*59))=1;
        elseif E(i,j)=='T'
            M(i,(j*59-1))=1;
        elseif E(i,j)=='C'
            M(i,(j*59-2))=1;
        elseif E(i,j)=='G'
            M(i,(j*59-3))=1;
        end
        
         for k=1:j
               if E(i,k)==E(i,j)
                   m=m+1;                     %在当前位置同一列和自己一样的加上一 碱基的含量
               end 
        end
           M(i,(j*47-43))=m/41;    %记录当前这一行的这一位置拥有同一信息的   
        
        if  E(i,j)=='A'&& E(i,j+1)=='A'
            M(i,(j*59-4))=-5.370;  M(i,(j*59-5))=2.900;  M(i,(j*59-6))=35.500;   M(i,(j*59-7))=0.000;   M(i,(j*59-8))=0.970;    M(i,(j*59-9))=-18.660;    M(i,(j*59-10))=0.0181;
            M(i,(j*59-11))=-1.200; M(i,(j*59-12))=1.900; M(i,(j*59-13))=66.510;  M(i,(j*59-14))=35.000; M(i,(j*59-15))=35.100;  M(i,(j*59-16))=3.900 ;
            M(i,(j*59-17))=0.960;  M(i,(j*59-18))=1.900; M(i,(j*59-19))=9.100;   M(i,(j*59-20))=24.000; M(i,(j*59-21))=11.400;  M(i,(j*59-22))=3.900;
            M(i,(j*59-23))=7.900;  M(i,(j*59-24))=0.970; M(i,(j*59-25))=16.300;  M(i,(j*59-26))=1.140;  M(i,(j*59-27))=1.020;   M(i,(j*59-28))=8.400;
            M(i,(j*59-29))=23.600; M(i,(j*59-30))=7.600; M(i,(j*59-31))=54.500;  M(i,(j*59-32))=5.370;  M(i,(j*59-33))=1.200;   M(i,(j*59-34))=8.000;
            M(i,(j*59-35))=21.900; M(i,(j*59-36))=26.000;  M(i,(j*59-37))=0.026; M(i,(j*59-38))=0.038;  M(i,(j*59-39))=0.020;   M(i,(j*59-40))=1.690;
            M(i,(j*59-41))=2.260;  M(i,(j*59-42))=7.650;
            
        elseif E(i,j)=='A'&& E(i,j+1)=='C'
            M(i,(j*59-4))=-10.510;   M(i,(j*59-5))=2.300;    M(i,(j*59-6))=33.100;   M(i,(j*59-7))=1.000;    M(i,(j*59-8))=0.130;   M(i,(j*59-9))=-13.100;    M(i,(j*59-10))=0.0854;
            M(i,(j*59-11))=-1.500;   M(i,(j*59-12))=1.300;   M(i,(j*59-13))=108.800; M(i,(j*59-14))=60.000;  M(i,(j*59-15))=31.500; M(i,(j*59-16))=4.600;
            M(i,(j*59-17))=4.520;    M(i,(j*59-18))=1.300 ;  M(i,(j*59-19))=6.500;   M(i,(j*59-20))=17.300;  M(i,(j*59-21))=19.800; M(i,(j*59-22))=1.300;
            M(i,(j*59-23))=5.400;    M(i,(j*59-24))=0.130;   M(i,(j*59-25))=5.400;   M(i,(j*59-26))=2.470;   M(i,(j*59-27))=1.430;  M(i,(j*59-28))=8.600;
            M(i,(j*59-29))=23.000 ;  M(i,(j*59-30))=14.600 ; M(i,(j*59-31))=97.730;  M(i,(j*59-32))=10.510 ; M(i,(j*59-33))=1.500;  M(i,(j*59-34))=9.400;
            M(i,(j*59-35))=25.500;   M(i,(j*59-36))=34.000;  M(i,(j*59-37))=0.036;   M(i,(j*59-38))=0.038 ;  M(i,(j*59-39))=0.023;  M(i,(j*59-40))=1.320;
            M(i,(j*59-41))=3.030;    M(i,(j*59-42))=8.930;
            
        elseif E(i,j)=='A'&& E(i,j+1)=='G'
            M(i,(j*59-4))=-6.780;  M(i,(j*59-5))=2.100;   M(i,(j*59-6))=30.600 ; M(i,(j*59-7))=1.000;   M(i,(j*59-8))=0.330;    M(i,(j*59-9))=-14.000;    M(i,(j*59-10))=0.0642;
            M(i,(j*59-11))=-1.500; M(i,(j*59-12))=1.600;  M(i,(j*59-13))=85.120; M(i,(j*59-14))=60.000; M(i,(j*59-15))=31.900;  M(i,(j*59-16))=3.400;
            M(i,(j*59-17))=0.790;  M(i,(j*59-18))=1.600;  M(i,(j*59-19))=7.800;  M(i,(j*59-20))=20.800; M(i,(j*59-21))=19.800;  M(i,(j*59-22))=3.400;
            M(i,(j*59-23))=6.700;  M(i,(j*59-24))=0.330;  M(i,(j*59-25))=14.200; M(i,(j*59-26))=2.470;  M(i,(j*59-27))=1.160;   M(i,(j*59-28))=6.100;
            M(i,(j*59-29))=16.100; M(i,(j*59-30))=8.200;  M(i,(j*59-31))=58.420; M(i,(j*59-32))=6.780;  M(i,(j*59-33))=1.500;   M(i,(j*59-34))=6.600;
            M(i,(j*59-35))=16.400; M(i,(j*59-36))=34.000; M(i,(j*59-37))=0.031;  M(i,(j*59-38))=0.037;  M(i,(j*59-39))=0.019;   M(i,(j*59-40))=1.460;
            M(i,(j*59-41))=2.030 ; M(i,(j*59-42))=7.080;
            
        elseif E(i,j)=='A'&& E(i,j+1)=='T'
            M(i,(j*59-4))=-6.570;  M(i,(j*59-5))=1.600;   M(i,(j*59-6))=59.200;   M(i,(j*59-7))=0.000;    M(i,(j*59-8))=0.580;   M(i,(j*59-9))=-15.010;    M(i,(j*59-10))=0.0542;
            M(i,(j*59-11))=-0.900; M(i,(j*59-12))=0.900;  M(i,(j*59-13))=72.290;  M(i,(j*59-14))=20.000;  M(i,(j*59-15))=29.300; M(i,(j*59-16))=5.900;
            M(i,(j*59-17))=6.820;  M(i,(j*59-18))=1.500;  M(i,(j*59-19))=8.600;   M(i,(j*59-20))=23.900;  M(i,(j*59-21))=11.400; M(i,(j*59-22))=2.400;
            M(i,(j*59-23))=6.300;  M(i,(j*59-24))=0.580;  M(i,(j*59-25))=10.000;  M(i,(j*59-26))=1.140;   M(i,(j*59-27))=0.730;  M(i,(j*59-28))=6.500;
            M(i,(j*59-29))=18.800; M(i,(j*59-30))=25.000; M(i,(j*59-31))=57.020;  M(i,(j*59-32))=6.570;   M(i,(j*59-33))=0.900;  M(i,(j*59-34))=5.600;
            M(i,(j*59-35))=15.200; M(i,(j*59-36))=26.000; M(i,(j*59-37))=0.033;   M(i,(j*59-38))=0.036;   M(i,(j*59-39))=0.022;  M(i,(j*59-40))=1.030;
            M(i,(j*59-41))=3.830;  M(i,(j*59-42))=9.070;
            
        elseif E(i,j)=='C'&& E(i,j+1)=='A'
            M(i,(j*59-4))=-6.570;  M(i,(j*59-5))=9.800;     M(i,(j*59-6))=37.700;   M(i,(j*59-7))=1.000;   M(i,(j*59-8))=1.040;    M(i,(j*59-9))=-9.450;    M(i,(j*59-10))=0.0752;
            M(i,(j*59-11))=-1.700; M(i,(j*59-12))=1.900;    M(i,(j*59-13))=64.920 ; M(i,(j*59-14))=60.000; M(i,(j*59-15))=37.300;  M(i,(j*59-16))=1.300 ;
            M(i,(j*59-17))=5.100;  M(i,(j*59-18))=1.900;    M(i,(j*59-19))=5.800;   M(i,(j*59-20))=12.900; M(i,(j*59-21))=19.800;  M(i,(j*59-22))=4.600;
            M(i,(j*59-23))=7.900;  M(i,(j*59-24))=1.040;    M(i,(j*59-25))=19.200;  M(i,(j*59-26))=2.470;  M(i,(j*59-27))=1.380;   M(i,(j*59-28))=7.400;
            M(i,(j*59-29))=19.300; M(i,(j*59-30))=10.900;   M(i,(j*59-31))=72.550;  M(i,(j*59-32))=6.570;  M(i,(j*59-33))=1.700;   M(i,(j*59-34))=8.200 ;
            M(i,(j*59-35))=21.000; M(i,(j*59-36))=34.000;   M(i,(j*59-37))=0.016;   M(i,(j*59-38))=0.025;  M(i,(j*59-39))=0.017;   M(i,(j*59-40))=1.070;
            M(i,(j*59-41))=1.780;  M(i,(j*59-42))=6.380;
            
        elseif E(i,j)=='C'&& E(i,j+1)=='C'
            M(i,(j*59-4))=-8.260;   M(i,(j*59-5))=6.100;   M(i,(j*59-6))=35.300;   M(i,(j*59-7))=2.000;     M(i,(j*59-8))=0.190;   M(i,(j*59-9))=-8.110;    M(i,(j*59-10))=0.0979;
            M(i,(j*59-11))=-2.300;  M(i,(j*59-12))=3.100;  M(i,(j*59-13))=99.310 ; M(i,(j*59-14))=130.000;  M(i,(j*59-15))=32.900; M(i,(j*59-16))=2.400;
            M(i,(j*59-17))=2.260 ;  M(i,(j*59-18))=3.100;  M(i,(j*59-19))=11.000;  M(i,(j*59-20))=26.600;   M(i,(j*59-21))=28.200; M(i,(j*59-22))=2.400;
            M(i,(j*59-23))=13.000;  M(i,(j*59-24))=0.190;  M(i,(j*59-25))=10.000;  M(i,(j*59-26))=3.800 ;   M(i,(j*59-27))=1.770;  M(i,(j*59-28))=6.700;
            M(i,(j*59-29))=15.600;  M(i,(j*59-30))=7.200;  M(i,(j*59-31))=54.710;  M(i,(j*59-32))=8.260;    M(i,(j*59-33))=2.100;  M(i,(j*59-34))=10.900;
            M(i,(j*59-35))=28.400;  M(i,(j*59-36))=42.000; M(i,(j*59-37))=0.026;   M(i,(j*59-38))=0.042;    M(i,(j*59-39))=0.019;  M(i,(j*59-40))=1.430;
            M(i,(j*59-41))=1.650;   M(i,(j*59-42))=8.040;
            
        elseif E(i,j)=='C'&& E(i,j+1)=='G'
            M(i,(j*59-4))=-9.610;   M(i,(j*59-5))=12.100;   M(i,(j*59-6))=31.300;  M(i,(j*59-7))=2.000;    M(i,(j*59-8))=0.520;    M(i,(j*59-9))=-10.030;    M(i,(j*59-10))=0.0597;
            M(i,(j*59-11))=-2.800;  M(i,(j*59-12))=3.600;   M(i,(j*59-13))=88.840; M(i,(j*59-14))=85.000;  M(i,(j*59-15))=36.100;  M(i,(j*59-16))=0.700 ;
            M(i,(j*59-17))=10.790;  M(i,(j*59-18))=3.600;   M(i,(j*59-19))=11.900; M(i,(j*59-20))=27.800;  M(i,(j*59-21))=28.200;  M(i,(j*59-22))=4.000;
            M(i,(j*59-23))=15.100;  M(i,(j*59-24))=0.520;   M(i,(j*59-25))=16.700; M(i,(j*59-26))=3.800;   M(i,(j*59-27))=2.090;   M(i,(j*59-28))=10.100;
            M(i,(j*59-29))=25.500;  M(i,(j*59-30))=8.900;   M(i,(j*59-31))=54.710; M(i,(j*59-32))=9.690;   M(i,(j*59-33))=2.800;   M(i,(j*59-34))=11.800;
            M(i,(j*59-35))=29.000;  M(i,(j*59-36))=42.000;  M(i,(j*59-37))=0.014;  M(i,(j*59-38))=0.026 ;  M(i,(j*59-39))=0.016;   M(i,(j*59-40))=1.080 ;
            M(i,(j*59-41))=2.000;   M(i,(j*59-42))=6.230;
            
        elseif E(i,j)=='C'&& E(i,j+1)=='T'
            M(i,(j*59-4))=-6.780;    M(i,(j*59-5))=2.100;    M(i,(j*59-6))=30.600 ; M(i,(j*59-7))=1.000;   M(i,(j*59-8))=0.330;     M(i,(j*59-9))=-14.000;    M(i,(j*59-10))=0.0628;
            M(i,(j*59-11))=-1.500 ;  M(i,(j*59-12))=1.600;   M(i,(j*59-13))=85.120; M(i,(j*59-14))=60.000; M(i,(j*59-15))=31.900 ;  M(i,(j*59-16))=3.400;
            M(i,(j*59-17))=0.790;    M(i,(j*59-18))=1.600;   M(i,(j*59-19))=7.800;  M(i,(j*59-20))=20.800; M(i,(j*59-21))=19.800;   M(i,(j*59-22))=3.400;
            M(i,(j*59-23))=6.700;    M(i,(j*59-24))=0.330;   M(i,(j*59-25))=14.200; M(i,(j*59-26))=2.470;  M(i,(j*59-27))=1.160;    M(i,(j*59-28))=6.100;
            M(i,(j*59-29))=16.100;   M(i,(j*59-30))=8.200;   M(i,(j*59-31))=85.970; M(i,(j*59-32))=6.780;  M(i,(j*59-33))=1.500;    M(i,(j*59-34))=6.600;
            M(i,(j*59-35))=16.400;   M(i,(j*59-36))=34.000;  M(i,(j*59-37))=0.031;  M(i,(j*59-38))=0.037;  M(i,(j*59-39))=0.019;    M(i,(j*59-40))=1.460;
            M(i,(j*59-41))=2.030;    M(i,(j*59-42))=7.080;
            
        elseif E(i,j)=='G'&& E(i,j+1)=='A'
            M(i,(j*59-4))=-9.810;    M(i,(j*59-5))=4.500;    M(i,(j*59-6))=39.600;   M(i,(j*59-7))=1.000 ;   M(i,(j*59-8))=0.980;     M(i,(j*59-9))=-13.480;    M(i,(j*59-10))=0.0623;
            M(i,(j*59-11))=-1.500;   M(i,(j*59-12))=1.600;   M(i,(j*59-13))=80.030;  M(i,(j*59-14))=60.000;  M(i,(j*59-15))=36.300;   M(i,(j*59-16))=3.400 ;
            M(i,(j*59-17))=3.180;    M(i,(j*59-18))=1.600 ;  M(i,(j*59-19))=5.600 ;  M(i,(j*59-20))=13.500;  M(i,(j*59-21))=19.800;   M(i,(j*59-22))=2.500;
            M(i,(j*59-23))=6.700;    M(i,(j*59-24))=0.980;   M(i,(j*59-25))=10.500 ; M(i,(j*59-26))=2.470;   M(i,(j*59-27))=1.460;    M(i,(j*59-28))=7.700;
            M(i,(j*59-29))=20.300;   M(i,(j*59-30))=8.800;   M(i,(j*59-31))=86.440;  M(i,(j*59-32))=9.810;   M(i,(j*59-33))=1.500;    M(i,(j*59-34))=8.800;
            M(i,(j*59-35))=23.500;   M(i,(j*59-36))=34.000;  M(i,(j*59-37))=0.025;   M(i,(j*59-38))=0.038;   M(i,(j*59-39))=0.020;    M(i,(j*59-40))=1.320    ;
            M(i,(j*59-41))=1.930;    M(i,(j*59-42))=8.560;
            
        elseif E(i,j)=='G'&& E(i,j+1)=='C'
            M(i,(j*59-4))=-14.590;  M(i,(j*59-5))=4.000;    M(i,(j*59-6))=38.400;    M(i,(j*59-7))=2.000;    M(i,(j*59-8))=0.730;     M(i,(j*59-9))=-11.080;    M(i,(j*59-10))=0.0506 ;
            M(i,(j*59-11))=-2.300;  M(i,(j*59-12))=3.100;   M(i,(j*59-13))=135.830;  M(i,(j*59-14))=85.000;  M(i,(j*59-15))=33.600;   M(i,(j*59-16))=4.000;
            M(i,(j*59-17))=8.280;   M(i,(j*59-18))=3.100;   M(i,(j*59-19))=11.100;   M(i,(j*59-20))=26.700;  M(i,(j*59-21))=28.200;   M(i,(j*59-22))=0.700;
            M(i,(j*59-23))=13.000;  M(i,(j*59-24))=0.730;   M(i,(j*59-25))=2.900;    M(i,(j*59-26))=3.800;   M(i,(j*59-27))=2.280;    M(i,(j*59-28))=11.100;
            M(i,(j*59-29))=28.400;  M(i,(j*59-30))=11.100 ; M(i,(j*59-31))=136.120;  M(i,(j*59-32))=14.590;  M(i,(j*59-33))=2.300;    M(i,(j*59-34))=10.500 ;
            M(i,(j*59-35))=26.400;  M(i,(j*59-36))=42.000;  M(i,(j*59-37))=0.025;    M(i,(j*59-38))=0.036 ;  M(i,(j*59-39))=0.026;    M(i,(j*59-40))=1.200;
            M(i,(j*59-41))=2.610;   M(i,(j*59-42))=9.530 ;
            
        elseif E(i,j)=='G'&& E(i,j+1)=='G'
            M(i,(j*59-4))=-8.260;   M(i,(j*59-5))=6.100;    M(i,(j*59-6))=35.300;    M(i,(j*59-7))=2.000;     M(i,(j*59-8))=0.190;    M(i,(j*59-9))=-8.110 ;    M(i,(j*59-10))= 0.0378;
            M(i,(j*59-11))=-2.300;  M(i,(j*59-12))=3.100;   M(i,(j*59-13))=99.310;   M(i,(j*59-14))=130.000;  M(i,(j*59-15))=32.900;  M(i,(j*59-16))=2.400;
            M(i,(j*59-17))=2.260;   M(i,(j*59-18))=3.100;   M(i,(j*59-19))=11.000;   M(i,(j*59-20))=26.600;   M(i,(j*59-21))=28.200;  M(i,(j*59-22))=2.400;
            M(i,(j*59-23))=13.000;  M(i,(j*59-24))=0.190;   M(i,(j*59-25))=10.000;   M(i,(j*59-26))=3.800;    M(i,(j*59-27))=1.770 ;  M(i,(j*59-28))=6.700;
            M(i,(j*59-29))=15.600;  M(i,(j*59-30))=7.200;   M(i,(j*59-31))=85.970 ;  M(i,(j*59-32))=8.260;    M(i,(j*59-33))=2.100;   M(i,(j*59-34))=10.900 ;
            M(i,(j*59-35))=28.400;  M(i,(j*59-36))=42.000;  M(i,(j*59-37))=0.026;    M(i,(j*59-38))=0.042 ;   M(i,(j*59-39))=0.019;   M(i,(j*59-40))=1.430;
            M(i,(j*59-41))=1.650;   M(i,(j*59-42))=8.040;
            
        elseif E(i,j)=='G'&& E(i,j+1)=='T'
            M(i,(j*59-4))=-10.510;  M(i,(j*59-5))=2.300;    M(i,(j*59-6))=33.100;   M(i,(j*59-7))=1.000;     M(i,(j*59-8))=0.130;    M(i,(j*59-9))=-13.100;    M(i,(j*59-10))=0.0425;
            M(i,(j*59-11))=-1.500;  M(i,(j*59-12))=1.300;   M(i,(j*59-13))=108.800; M(i,(j*59-14))=60.000;   M(i,(j*59-15))=31.500;  M(i,(j*59-16))=4.600;
            M(i,(j*59-17))=4.520 ;  M(i,(j*59-18))=1.300;   M(i,(j*59-19))=6.500;   M(i,(j*59-20))=17.300;   M(i,(j*59-21))=19.800;  M(i,(j*59-22))=1.300;
            M(i,(j*59-23))=5.400;   M(i,(j*59-24))=0.130;   M(i,(j*59-25))=5.400;   M(i,(j*59-26))=2.470;    M(i,(j*59-27))=1.430;   M(i,(j*59-28))=8.600;
            M(i,(j*59-29))=23.000;  M(i,(j*59-30))=14.600;  M(i,(j*59-31))=97.730;  M(i,(j*59-32))=10.510;   M(i,(j*59-33))=1.500;   M(i,(j*59-34))=9.400;
            M(i,(j*59-35))=25.500;  M(i,(j*59-36))=34.000;  M(i,(j*59-37))=0.036;   M(i,(j*59-38))=0.038;    M(i,(j*59-39))=0.023;   M(i,(j*59-40))=1.320 ;
            M(i,(j*59-41))=3.030;   M(i,(j*59-42))=8.930;
            
        elseif E(i,j)=='T'&& E(i,j+1)=='A'
            M(i,(j*59-4))=-3.820;   M(i,(j*59-5))=6.300;    M(i,(j*59-6))=31.600;   M(i,(j*59-7))=0.000;    M(i,(j*59-8))=0.730;    M(i,(j*59-9))=-11.850;    M(i,(j*59-10))=0.0673;
            M(i,(j*59-11))=-0.900;  M(i,(j*59-12))=1.500;   M(i,(j*59-13))=50.110;  M(i,(j*59-14))=20.000;  M(i,(j*59-15))=37.800;  M(i,(j*59-16))=2.500;
            M(i,(j*59-17))=0.420;   M(i,(j*59-18))=0.900;   M(i,(j*59-19))=6.000;   M(i,(j*59-20))=16.900;  M(i,(j*59-21))=11.400;  M(i,(j*59-22))=5.900;
            M(i,(j*59-23))=3.800;   M(i,(j*59-24))=0.730;   M(i,(j*59-25))=24.700;  M(i,(j*59-26))=1.140;   M(i,(j*59-27))=0.600;   M(i,(j*59-28))=6.300;
            M(i,(j*59-29))=18.500;  M(i,(j*59-30))=12.500;  M(i,(j*59-31))=36.730;  M(i,(j*59-32))= 3.820;  M(i,(j*59-33))=0.900;   M(i,(j*59-34))=6.600;
            M(i,(j*59-35))=18.400;  M(i,(j*59-36))=26.000;  M(i,(j*59-37))=0.017;   M(i,(j*59-38))=0.018;   M(i,(j*59-39))=0.016;   M(i,(j*59-40))=0.720;
            M(i,(j*59-41))=1.200;   M(i,(j*59-42))= 6.230;
             
        elseif E(i,j)=='T'&& E(i,j+1)=='C'
            M(i,(j*59-4))=-9.810;    M(i,(j*59-5))=4.500;    M(i,(j*59-6))=39.600;    M(i,(j*59-7))=1.000;     M(i,(j*59-8))=0.980;    M(i,(j*59-9))=-13.480;    M(i,(j*59-10))=0.0605;
            M(i,(j*59-11))=-1.500 ;  M(i,(j*59-12))=1.600;   M(i,(j*59-13))=80.030;   M(i,(j*59-14))=60.000;   M(i,(j*59-15))=36.300;  M(i,(j*59-16))=3.400;
            M(i,(j*59-17))=3.180;    M(i,(j*59-18))=1.600;   M(i,(j*59-19))=5.600;    M(i,(j*59-20))=13.500;   M(i,(j*59-21))=19.800;  M(i,(j*59-22))=2.500;
            M(i,(j*59-23))=6.700;    M(i,(j*59-24))=0.980;   M(i,(j*59-25))=10.500;   M(i,(j*59-26))=2.470;    M(i,(j*59-27))=1.460;   M(i,(j*59-28))=7.700;
            M(i,(j*59-29))=20.300;   M(i,(j*59-30))=8.800;   M(i,(j*59-31))=86.440;   M(i,(j*59-32))=9.810;    M(i,(j*59-33))=1.500;   M(i,(j*59-34))=8.800;
            M(i,(j*59-35))=23.500;   M(i,(j*59-36))=34.000;  M(i,(j*59-37))=0.025;    M(i,(j*59-38))=0.038;    M(i,(j*59-39))=0.020;   M(i,(j*59-40))=1.320;
            M(i,(j*59-41))=1.930;    M(i,(j*59-42))=8.560;
 
        elseif E(i,j)=='T'&& E(i,j+1)=='G'
            M(i,(j*59-4))=-6.570;   M(i,(j*59-5))=9.800;    M(i,(j*59-6))=37.700;   M(i,(j*59-7))=1.000;    M(i,(j*59-8))=1.040;    M(i,(j*59-9))=-9.450;    M(i,(j*59-10))=0.0287 ;
            M(i,(j*59-11))=-1.700;  M(i,(j*59-12))=1.900;   M(i,(j*59-13))=64.920;  M(i,(j*59-14))=60.000;  M(i,(j*59-15))=37.300;  M(i,(j*59-16))=1.300;
            M(i,(j*59-17))=5.100;   M(i,(j*59-18))=1.900;   M(i,(j*59-19))=5.800;   M(i,(j*59-20))=12.900;  M(i,(j*59-21))=19.800;  M(i,(j*59-22))=4.600;
            M(i,(j*59-23))=7.900;   M(i,(j*59-24))=1.040;   M(i,(j*59-25))=19.200;  M(i,(j*59-26))=2.470;   M(i,(j*59-27))=1.380;   M(i,(j*59-28))=7.400;
            M(i,(j*59-29))=19.300;  M(i,(j*59-30))=10.900;  M(i,(j*59-31))=58.420;  M(i,(j*59-32))=6.570;   M(i,(j*59-33))=1.700;   M(i,(j*59-34))=8.200;
            M(i,(j*59-35))=21.000;  M(i,(j*59-36))=34.000;  M(i,(j*59-37))=0.016;   M(i,(j*59-38))=0.025;   M(i,(j*59-39))=0.017;   M(i,(j*59-40))= 1.070;
            M(i,(j*59-41))=1.780;   M(i,(j*59-42))=6.380;
            
        elseif E(i,j)=='T'&& E(i,j+1)=='T'
            M(i,(j*59-4))=-5.370;    M(i,(j*59-5))=2.900;    M(i,(j*59-6))=35.500;   M(i,(j*59-7))=0.000;     M(i,(j*59-8))=0.970;    M(i,(j*59-9))=-18.660;    M(i,(j*59-10))=0.0406;
            M(i,(j*59-11))=-1.200;   M(i,(j*59-12))=1.900 ;  M(i,(j*59-13))=66.510;  M(i,(j*59-14))=35.000;   M(i,(j*59-15))=35.100;  M(i,(j*59-16))=3.900;
            M(i,(j*59-17))=0.960;    M(i,(j*59-18))=1.900;   M(i,(j*59-19))=9.100;   M(i,(j*59-20))=24.000;   M(i,(j*59-21))=11.400;  M(i,(j*59-22))= 3.900;
            M(i,(j*59-23))=7.900;    M(i,(j*59-24))=0.970;   M(i,(j*59-25))=16.300;  M(i,(j*59-26))=1.140;    M(i,(j*59-27))=1.020;   M(i,(j*59-28))=8.400;
            M(i,(j*59-29))=23.600;   M(i,(j*59-30))=7.600;   M(i,(j*59-31))=54.500;  M(i,(j*59-32))=5.370;    M(i,(j*59-33))=1.200;   M(i,(j*59-34))=8.000;
            M(i,(j*59-35))= 21.900;  M(i,(j*59-36))=26.000;  M(i,(j*59-36))=0.026;   M(i,(j*59-38))=0.038;    M(i,(j*59-39))= 0.000;  M(i,(j*59-40))=1.690;
            M(i,(j*59-41))=2.260;    M(i,(j*59-42))=7.650;
            
        end
%%%%%%% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
         h1=0;h2=0;h3=0;h4=0;h5=0;h6=0;h7=0;h8=0;h9=0;h10=0;h11=0;h12=0;
        if E(i,j)=='A'&&(j<39)
            if E(i,j+1)=='A'
                if E(i,j+2)=='G'
                    h1=-0.506; h2=-0.257; h3=-0.577; h4=0.155;   h5=-0.260;  h6=-0.260; 
                    h7=0.193 ; h8=0.493;  h9=-1.000; h10=-1.000; h11=0.169;  h12=-0.179; 
                elseif E(i,j+2)=='A'
                    h1=-2.087; h2=-2.745; h3=-1.732; h4=-2.349;  h5=-2.744;  h6=-2.744;
                    h7=2.274; h8=2.118;  h9=-1.000; h10=-1.000; h11=-2.342;  h12=2.386;
                elseif E(i,j+2)=='C'
                    h1=-1.509 ; h2=-1.354 ; h3=-0.577; h4=-0.561;  h5=-1.363;  h6=-1.363 ;
                    h7=1.105;   h8=1.516;   h9=-1.000; h10=-1.000; h11=-0.555; h12=0.548;
                elseif E(i,j+2)=='T'
                    h1=-2.126; h2=-2.585; h3=-1.732; h4=-1.991;  h5=-2.591;   h6=-2.591;
                    h7=2.141;  h8=2.158;  h9=-1.000; h10=-1.000; h11=-2.004;  h12=2.032;                   
                end                 
                
            elseif E(i,j+1)=='T'
                if E(i,j+2)=='G'                  
                    h1=1.229; h2=1.348;   h3=-0.577; h4=0.870;   h5=1.358;   h6=1.358;
                    h7=-1.112; h8=-1.215; h9=-1.000; h10=-1.000; h11=0.893;  h12=-0.894 ;
                elseif E(i,j+2)=='A'                   
                    h1=1.615; h2=0.572;   h3=-1.732; h4=-0.978;  h5=0.584;  h6=0.584;
                    h7=-0.491; h8=-1.585; h9=-1.000; h10=-1.000; h11=-0.990;  h12=0.988;
                elseif E(i,j+2)=='C'  
                    h1=-0.737; h2=-0.391; h3=-0.577; h4=0.214;  h5=-0.397;   h6=-0.397;
                    h7=0.307; h8=0.727;  h9=-1.000; h10=-1.000; h11=0.217;  h12=-0.227 ;
                elseif E(i,j+2)=='T'   
                    h1=-2.126; h2=-2.585; h3=-1.732; h4=-1.991;  h5=-2.591;  h6=-2.591;
                    h7=2.141; h8=2.158;  h9=-1.000; h10=-1.000; h11=-2.004;  h12=2.032;
                end      
            elseif E(i,j+1)=='G'
                if E(i,j+2)=='G'
                    h1=-0.313; h2=-0.070; h3=0.577; h4=0.274;  h5=-0.070;  h6=-0.070;
                    h7=0.039; h8=0.300;  h9=1.000; h10=1.000; h11=0.266 ;  h12=-0.275 ;  
                elseif E(i,j+2)=='A'
                    h1=0.381; h2=-0.150; h3=-0.577; h4=-0.740;  h5=-0.158;  h6=-0.158;
                    h7=0.109; h8=-0.389;  h9=1.000; h10=1.000; h11=-0.748;  h12=0.743;
                elseif E(i,j+2)=='C'
                    h1=0.304; h2=0.920; h3=0.577; h4=1.287;  h5=0.911;  h6=0.911;
                    h7=-0.753; h8=-0.313;  h9=1.000; h10=1.000; h11=1.280;  h12=-1.272;
                elseif E(i,j+2)=='T'
                    h1=-1.354; h2=-0.685; h3=-0.577; h4=0.453;  h5=-0.676;  h6=-0.676;
                    h7=0.536; h8=1.357;  h9=1.000; h10=1.000; h11=0.459;  h12=-0.466;
                end    
            elseif E(i,j+1)=='C'
               if E(i,j+2)=='G'
                    h1=-0.121; h2=0.064; h3=0.577; h4=0.274;   h5=0.065;   h6=0.065;
                    h7=-0.074; h8=0.107;  h9=1.000; h10=1.000; h11=0.266;  h12=-0.275;             
                elseif E(i,j+2)=='A'
                    h1=0.111; h2=0.171; h3=-0.577; h4=0.155;  h5=0.164;  h6=0.164;
                    h7=-0.153; h8=-0.123;  h9=1.000; h10=1.000; h11=0.169;  h12=-0.179;         
                elseif E(i,j+2)=='C'
                    h1=-0.121; h2=0.064; h3=0.577; h4=0.274;  h5=0.071;  h6=0.071;
                    h7=-0.078; h8=0.107; h9=1.000; h10=1.000; h11=0.266;  h12=-0.275;               
                elseif E(i,j+2)=='T'
                    h1=-1.354; h2=-0.685; h3=-0.577; h4=0.453;  h5=-0.676;  h6=-0.676;
                    h7=0.536;  h8=1.357;  h9=1.000;  h10=1.000; h11=0.459;  h12=-0.466;                   
                end    
            end
        elseif E(i,j)=='T'&&(j<39)
            if E(i,j+1)=='A'
               if E(i,j+2)=='G'            
                    h1=0.882; h2=-0.097; h3=-0.577; h4=-1.276;  h5=-0.097;  h6=-0.097;
                    h7=0.062; h8=-0.880;  h9=-1.000; h10=-1.000; h11=-1.280;  h12=1.285;
                elseif E(i,j+2)=='A'                    
                    h1=0.689; h2=-0.284; h3=-1.732; h4=-1.395;  h5=-0.275;  h6=-0.275;
                    h7=0.206; h8=-0.692;  h9=-1.000; h10=-1.000; h11=-1.376;  h12=1.384;
                elseif E(i,j+2)=='C'
                    h1=0.342; h2=-0.070; h3=-0.577; h4=-0.561;  h5=-0.062;  h6=-0.062;
                    h7=0.031; h8=-0.351;  h9=-1.000; h10=-1.000; h11=-0.555;  h12=0.548;
                elseif E(i,j+2)=='T'                    
                    h1=1.615; h2=0.572; h3=-1.732; h4=-0.978;  h5=0.584;  h6=0.584;
                    h7=-0.491; h8=-1.585;  h9=-1.000; h10=-1.000; h11=-0.990;  h12=0.988;
                end    
            elseif E(i,j+1)=='T'
                if E(i,j+2)=='G'                    
                    h1=0.265; h2=-0.231; h3=-0.577; h4=-0.740;  h5=-0.226;  h6=-0.226 ;
                    h7=0.166; h8=-0.275;  h9=-1.000; h10=-1.000; h11=-0.748;  h12=0.743;
                elseif E(i,j+2)=='A'
                    h1=0.689; h2=-0.284; h3=-1.732; h4=-1.395;  h5=-0.275;  h6=-0.275 ;
                    h7=0.206 ; h8=-0.692;  h9=-1.000; h10=-1.000; h11=-1.376 ;  h12=1.384;
                elseif E(i,j+2)=='C'
                    h1=-0.159; h2=-0.605; h3=-0.577; h4=-0.918;  h5=-0.600;  h6=-0.600;
                    h7=0.474; h8=0.146;  h9=-1.000; h10=-1.000; h11=-0.893 ;  h12=0.890;
                elseif E(i,j+2)=='T'                   
                    h1=-2.087; h2=-2.745; h3=-1.732; h4=-2.349;  h5=-2.744;  h6=-2.744;
                    h7=-2.615; h8=2.118;  h9=-1.000; h10=-1.000; h11=-2.342;  h12=2.386;
                end    
            elseif E(i,j+1)=='G'
                if E(i,j+2)=='G'
                    h1=-1.856; h2=-1.140; h3=0.577; h4=0.274;  h5=-1.139;  h6=-1.139;
                    h7=0.917; h8=1.876;  h9=1.000; h10=1.000; h11=0.266;  h12=-0.275;
                elseif E(i,j+2)=='A'
                    h1=1.730; h2=1.348; h3=-0.577; h4=0.274;  h5=1.348;  h6=1.348;
                    h7=4.522; h8=-1.696;  h9=1.000; h10=1.000; h11=0.266;  h12=-0.275;
                elseif E(i,j+2)=='C'
                    h1=0.766; h2=0.839; h3=0.577; h4=0.572;  h5=0.842;  h6=0.842;
                    h7=-0.702; h8=-0.767;  h9=1.000; h10=1.000; h11=0.555;  h12=-0.562;
                elseif E(i,j+2)=='T'
                    h1=0.111; h2=0.171; h3=-0.577; h4=0.155;  h5=0.164;  h6=0.164;
                    h7=-0.153; h8=-0.123;  h9=1.000; h10=1.000; h11=0.169;  h12=-0.179;
                end    
            elseif E(i,j+1)=='C'
                if E(i,j+2)=='G'                    
                    h1=0.111; h2=1.000; h3=0.577; h4=1.645;  h5=1.012;  h6=1.012;
                    h7=-0.834; h8=-0.123;  h9=1.000; h10=1.000; h11=1.666;  h12=-1.646;
                elseif E(i,j+2)=='A'
                    h1=1.730; h2=1.348; h3=-0.577; h4=0.274;  h5=1.348;  h6=1.348;
                    h7=-1.103; h8=-1.696;  h9=1.000; h10=1.000; h11=0.266;  h12=-0.275;
                elseif E(i,j+2)=='C'
                    h1=0.265; h2=-0.097; h3=0.577; h4=-0.501;  h5=-0.103;  h6=-0.103;
                    h7=0.066; h8=-0.275 ;  h9=1.000; h10=1.000; h11=-0.507;  h12=0.499 ;
                elseif E(i,j+2)=='T'
                    h1=0.381; h2=-0.150; h3=-0.577; h4=-0.740;  h5=-0.158;  h6=-0.158;
                    h7=0.109; h8=-0.389;  h9=1.000; h10=1.000; h11=-0.748;  h12=0.743;
                end    
            end
        elseif E(i,j)=='G'&&(j<39)
            if E(i,j+1)=='A'
                if E(i,j+2)=='G'
                    h1=0.419; h2=0.438; h3=0.577; h4=0.274;  h5=0.427;  h6=0.427;
                    h7=-0.365; h8=-0.427;  h9=-1.000; h10=-1.000; h11=0.266;  h12=-0.275;
                elseif E(i,j+2)=='A'
                    h1=-0.159; h2=-0.605; h3=-0.577; h4=-0.918;  h5=-0.600;  h6=-0.600;
                    h7=0.474; h8=0.146;  h9=-1.000; h10=-1.000; h11=-0.893;  h12=0.890;
                elseif E(i,j+2)=='C'
                    h1=0.034; h2=0.171; h3=0.577; h4=0.274;  h5=0.178;  h6=0.178;
                    h7=-0.165; h8=-0.046;  h9=-1.000; h10=-1.000; h11=0.266;  h12=-0.275;
                elseif E(i,j+2)=='T'                    
                    h1=-0.737; h2=-0.391; h3=-0.577; h4=0.214;  h5=-0.397;  h6=-0.397;
                    h7=0.307; h8=0.727;  h9=-1.000; h10=-1.000; h11=0.217;  h12=-0.227;
                end    
            elseif E(i,j+1)=='T'
                if E(i,j+2)=='G'
                    h1=0.496; h2=0.786; h3=0.577; h4=0.810;  h5=0.773;  h6=0.773;
                    h7=-0.646; h8=-0.503;  h9=-1.000; h10=-1.000; h11=0.797;  h12=-0.800;
                elseif E(i,j+2)=='A'
                    h1=0.342; h2=-0.070; h3=-0.577; h4=-0.561;  h5=-0.062;  h6=-0.062;
                    h7=0.031; h8=-0.351;  h9=-1.000; h10=-1.000; h11=-0.555;  h12=0.548;
                elseif E(i,j+2)=='C'
                    h1=0.034; h2=0.171; h3=0.577; h4=0.274;  h5=0.178;  h6=0.178;
                    h7=-0.165; h8=-0.046;  h9=-1.000; h10=-1.000; h11=0.266;  h12=-0.275;
                elseif E(i,j+2)=='T' 
                    h1=-1.509; h2=-1.354; h3=-0.577; h4=-0.561;  h5=-1.363;  h6=-1.363;
                    h7=1.105; h8=1.516;  h9=-1.000; h10=-1.000; h11=-0.555;  h12=0.548;
                end    
            elseif E(i,j+1)=='G'
                if E(i,j+2)=='G'
                    h1=0.072; h2=0.358; h3=1.732; h4=0.572;  h5=0.345;  h6=0.345;
                    h7=-0.300; h8=-0.084;  h9=1.000; h10=1.000; h11=0.555 ;  h12=-0.562;
                elseif E(i,j+2)=='A'                     
                    h1=0.265; h2=-0.097; h3=0.577; h4=-0.501;  h5=-0.103;  h6=-0.103;
                    h7=0.066; h8=-0.275;  h9=1.000; h10=1.000; h11=-0.507;  h12=0.499;
                elseif E(i,j+2)=='C'
                    h1=1.036; h2=2.097; h3=1.732; h4=2.479;  h5=2.089;  h6=2.089;
                    h7=-1.687; h8=-1.029;  h9=1.000; h10=1.000; h11=2.487;  h12=-2.433;
                elseif E(i,j+2)=='T'
                    h1=-0.121; h2=0.064; h3=0.577; h4=0.274;  h5=0.071;  h6=0.071;
                    h7=-0.078; h8=0.107;  h9=1.000; h10=1.000; h11=0.266;  h12=-0.275;
                end    
            elseif E(i,j+1)=='C'
                if E(i,j+2)=='G'
                    h1=-0.468; h2=0.385; h3=1.732; h4=1.287;  h5=0.379;  h6=0.379;
                    h7=-0.326; h8=0.455;  h9=1.000; h10=1.000; h11=1.280;  h12=-1.272;
                elseif E(i,j+2)=='A'
                    h1=0.766; h2=0.839; h3=0.577; h4=0.572;  h5=0.842;  h6=0.842;
                    h7=-0.702; h8=-0.767;  h9=1.000; h10=1.000; h11=0.555;  h12=-0.562;
                elseif E(i,j+2)=='C'
                    h1=1.036; h2=2.097; h3=1.732; h4=2.479;  h5=2.089;  h6=2.089;
                    h7=-1.687; h8=-1.029;  h9=1.000; h10=1.000 ; h11=2.487;  h12=-2.433;
                elseif E(i,j+2)=='T'
                    h1=0.304; h2=0.920; h3=0.577; h4=1.287;  h5=0.911;  h6=0.911;
                    h7=-0.753; h8=-0.313;  h9=1.000; h10=1.000; h11=1.280;  h12=-1.272;
                end    
            end
        elseif E(i,j)=='C'&&(j<39)
            if E(i,j+1)=='A'
               if E(i,j+2)=='G'
                    h1=1.576; h2=0.920; h3=0.577; h4=-0.322;  h5=0.920;  h6=0.920;
                    h7=-0.762; h8=-1.549;  h9=-1.000; h10=-1.000; h11=-0.314 ;  h12=0.304;
                elseif E(i,j+2)=='A'
                    h1=0.265; h2=-0.231; h3=-0.577; h4=-0.740;  h5=-0.226;  h6=-0.226;
                    h7=0.166; h8=-0.275;  h9=-1.000; h10=-1.000; h11=-0.748;  h12=0.743;
                elseif E(i,j+2)=='C'
                    h1=0.496; h2=0.786; h3=0.577; h4=0.810;  h5=0.773;  h6=0.773;
                    h7=-0.646; h8=-0.503;  h9=-1.000; h10=-1.000; h11=0.797;  h12=-0.800;
                elseif E(i,j+2)=='T'
                    h1=1.229; h2=1.348; h3=-0.577; h4=0.870;  h5=1.358;  h6=1.358;
                    h7=-1.112; h8=-1.215;  h9=-1.000; h10=-1.000; h11=0.893;  h12=-0.894;
                end    
            elseif E(i,j+1)=='T'
                if E(i,j+2)=='G'
                    h1=1.576; h2=0.920; h3=0.577; h4=-0.322;  h5=0.920;  h6=0.920;
                    h7=-0.762; h8=-1.549;  h9=-1.000; h10=-1.000; h11=-0.314;  h12=0.304;
                elseif E(i,j+2)=='A'
                    h1=0.882; h2=-0.097; h3=-0.577; h4=-1.276;  h5=-0.097;  h6=-0.097;
                    h7=0.062; h8=-0.880;  h9=-1.000; h10=-1.000; h11=-1.280;  h12=1.285;
                elseif E(i,j+2)=='C'
                    h1=0.419; h2=0.438; h3=0.577; h4=0.274;  h5=0.427;  h6=0.427;
                    h7=-0.365; h8=-0.427;  h9=-1.000; h10=-1.000; h11=0.266;  h12=-0.275;
                elseif E(i,j+2)=='T'
                    h1=-0.506; h2=-0.257; h3=-0.577; h4=0.155;  h5=-0.260;  h6=-0.260;
                    h7=0.193; h8=0.493;  h9=-1.000; h10=-1.000; h11=0.169;  h12=-0.179;
                end    
            elseif E(i,j+1)=='G'
                if E(i,j+2)=='G'   
                    h1=-0.969 ; h2=-0.712; h3=1.732; h4=-0.084;  h5=-0.705;  h6=-0.705;
                    h7=0.558; h8=0.962;  h9=1.000; h10=1.000; h11=-0.072;  h12=0.062;
                elseif E(i,j+2)=='A'
                    h1=0.111; h2=1.000; h3=0.577; h4=1.645;  h5=1.012;  h6=1.012;
                    h7=-0.834; h8=-0.123;  h9=1.000; h10=1.000; h11=1.666;  h12=-1.646;
                elseif E(i,j+2)=='C'                    
                    h1=-0.468; h2=0.385; h3=1.732; h4=1.287;  h5=0.379;  h6=0.379;
                    h7=-0.326; h8=0.455;  h9=1.000; h10=1.000; h11=1.280;  h12=-1.272;
                elseif E(i,j+2)=='T'
                    h1=-0.121; h2=0.064; h3=0.577; h4=0.274;  h5=0.065;  h6=0.065;
                    h7=-0.074; h8=0.107;  h9=1.000; h10=1.000; h11=0.266;  h12=-0.275;
                end    
            elseif E(i,j+1)=='C'
                if E(i,j+2)=='G'
                    h1=-0.969; h2=-0.712; h3=1.732; h4=-0.084;  h5=-0.705;  h6=-0.705;
                    h7=0.558; h8=0.962;  h9=1.000; h10=1.000; h11=-0.072;  h12=0.062;
                elseif E(i,j+2)=='A'
                    h1=-1.856; h2=-1.140; h3=0.577; h4=0.274;  h5=-1.139;  h6=-1.139;
                    h7=0.917; h8=1.876;  h9=1.000; h10=1.000; h11=0.266;  h12=-0.275;
                elseif E(i,j+2)=='C'                    
                    h1=0.072; h2=0.358; h3=1.732; h4=0.572;  h5=0.345;  h6=0.345;
                    h7=-0.300; h8=-0.084;  h9=1.000; h10=1.000; h11=0.555 ;  h12=-0.562;
                elseif E(i,j+2)=='T' 
                    h1=-0.313; h2=-0.070; h3=0.577; h4=0.274;  h5=-0.070;  h6=-0.070;
                    h7=0.039; h8=0.300;  h9=1.000; h10=1.000; h11=0.266;  h12=-0.275;
                end    
            end
        end
        M(i,j*59-47)=h1;
        M(i,j*59-48)=h2;
        M(i,j*59-49)=h3;
        M(i,j*59-50)=h4;
        M(i,j*59-51)=h5;
        M(i,j*59-52)=h6;
        M(i,j*59-53)=h7;
        M(i,j*59-54)=h8;
        M(i,j*59-55)=h9;
        M(i,j*59-56)=h10;
        M(i,j*59-57)=h11;
        M(i,j*59-58)=h12;  
        
        if E(i,j)=='G'||E(i,j)=='A'
            M(i,j*59-44)=1;
        end
        
        if E(i,j)=='T'||E(i,j)=='A'
            M(i,j*59-45)=1;
        end
        
        if E(i,j)=='A'||E(i,j)=='C'
            M(i,j*59-46)=1;
        end
    end
     if E(i,41)=='A'
        M(i,(59*41))=1;
    elseif E(i,41)=='T'
        M(i,(59*41-1))=1;
    elseif E(i,41)=='C'
        M(i,(59*41-2))=1;
    elseif E(i,41)=='G'
        M(i,(59*41-3))=1;
    end
end
  

%将ATCG的每一位编译成新的四位数值
a=ones(1554,1);
b=zeros(1554,1);
c=[a;b];
M=[c,M];

