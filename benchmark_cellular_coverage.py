"""High-dimensional composite cellular coverage/capacity benchmark.

The oracle is derived from the public CCO-in-O-RAN radio-propagation dataset:

    https://github.com/Ryandry1st/CCO-in-ORAN
    commit bb9947ec5f5172f139e238be81aeb072acfc3b99

There are 15 independently configurable sectors.  The first 15 normalized
inputs set electrical downtilt in [0, 10] degrees, and the last 15 set transmit
power in [30, 50] dBm, giving a 30-dimensional scientific design problem.

To keep one copied Python file sufficient on an offline cluster, this script
embeds the published received-power maps at a fixed 4x4 spatial raster.  The
16 locations span both map axes at source indices [20, 87, 153, 220], or about
[-500, -165, 165, 500] metres.  Each embedded value is taken from the original
241x241 map for every integer downtilt and sector; no fitted surrogate is used.
Continuous downtilts use the source implementation's linear interpolation.

Composite response (32 intermediates -> 2 objectives):

* h_1..h_16 are serving-cell RSRP values in dBm.
* h_17..h_32 are aggregate non-serving-cell interference values in dBm.
* objective 1 is mean soft weak-coverage violation relative to -80 dBm.
* objective 2 is mean soft over-coverage violation for interference within
  6 dB of the serving signal, gated to locations that have coverage.

Both objectives are minimized.  These are differentiable relaxations of the
two masks and sigmoid penalties in the released problem_formulation.py.
The spatial map is therefore collapsed through thresholds, sigmoids, products,
and a mean.  That nonlinear many-to-two reduction is precisely the information
loss that the composite high-dimensional methods are intended to avoid.

Run the paper protocol (10 trials, 20 initial points, 120 total evaluations):

    python benchmark_cellular_coverage.py

For a smoke test:

    python benchmark_cellular_coverage.py --quick
"""

from __future__ import annotations

import base64
import zlib

import numpy as np
import torch

from benchmark_common import BenchmarkProblem, run_benchmark


N_SECTORS = 15
RASTER_SIDE = 4
N_LOCATIONS = RASTER_SIDE * RASTER_SIDE
DIM = 2 * N_SECTORS
WEAK_COVERAGE_THRESHOLD_DBM = -80.0
OVER_COVERAGE_MARGIN_DB = 6.0
SOFT_GATE_SCALE_DB = 2.0
MIN_INTERFERENCE_POWER_WATT = 1.0e-24

# Shape after decoding: [11 integer downtilts, 16 locations, 15 sectors].
# Values are float32 received powers normalized to 0 dBm transmit power.
_EMBEDDED_POWER_MAP_B85 = (
    "c-keJXINBM*M^5W^g0w7rT_y7N}IyaQB*|KsIhA_8hd|j*kdQr*jwy1_LkUNEc>>^-eW~YjbLve3aH<f@5ghkKQm{}S!dsCJ<oN`"
    "k>P$!Uz1AoJ_jbP$!4fWHm$w<S=w9A&<=Vs7T*@523ql*ZRNuUTQ>b^q5P<sBS-ByVDhAJdL~orc(AhATBSL2ZP2EKa%xpLYv(x9"
    "_D2nxtVllOJ`#6aEnK{3!P(Oen-qVhCgf6YoSy36$5m;|HCvKPJs3YGilTZUj9DJX<HuQCO7NlNg&!YI#nU}r!ELja^%GjK=TSTj"
    "eh($_TmoY@1+i*XC<*orxx6+PLyU^Eekmk2bffuE7glz$uyef^Ge^4<tTNKd_Jg>#C6@fZw2a&3z@AJMr%siL74IJiuY)F9oiveV"
    "R?_-P5*>75B((^j#xJ&9`o~D?Lr(nSq{GAPP36pTG5O{{V%S|Jvu-H)cfJMHH7kc&*^u{d2zlG|#2+ga+D7&ygr>6jRUnS79of<H"
    "nQ%Q>C^lX@EfTMK^G!ulRM>Qhqfzw|QE{h4#5C0~a8!jjSnkTLd6`^o8Hw$>n?m)*3ELVPZajCTpiTxwbu*}_&~S8X4zCn?ZcKV1"
    "lII&3yG_M~k+!5{Df#=zIk97+6~o_tObgLaI#ErROOHh4@(^OO%%n$}nb^*YZP%^z>FY*|gGT<DUM31WV~NpLqo2jdz{}2PW*XSN"
    "?2gzz<Ehy3B9IQ@`YIh=_FDK2SJT<=nuu<$X3-rRGNVjPyc165MNgLa{3lvm(Brm9LErV^ESPNJWUVrBcw`VyVsDCtZ6Ax^Y+D95"
    "4&YgqJr71&81poW4Ff-mFLx@$EjM`|)s+|{tbEj_@w|yWk1A8>RbS5!gB)2F5s&|Y1f~ttGG6yWe5?8u8JuwQB{RUp);vAl18w-<"
    "FG=ie9L5h;H#F}8sW_ZY!84btIxba%i<1V^Uvd0<$3*H5H&!1?!!BQm`%`-soJnBQWi?BhD;N>u$HxaI>i7688vSl&u%n(O2d#wn"
    "O2y%sio9WEqE}NR&+0ql_$i2WZpkFqyeh6v{6kcnSFn0P0DXHqBjSu~zO7|zS{)`;tIMozN{%e}Eb@v?IIO4?O<w3|-_k;xM0<K~"
    "QuFzXndG-#WEF&BO-SU2JQW#|-<O;1u}FTG>ay9i$&bd8-{bA{%+y^M9Y$I?KFdn^Fk9M7evA5>>A1;`*?l~jE%`k>!GmU!-EUgG"
    "TtU=tAq<!NCd{`-bkT6Wb|h2QJrp|~ER2)<E;ZQVxzUeQ$?t%tI&L+KtI{-`4Q_gOjFR79OLR;-9Ls#k@1b7a5bMW!$#1LiU&OWk"
    "8rn#Hn*_$wt*4%%vk55L1u|(*2!(bH8MP#rZnIR3lKj@rcVk^$XPQZV?Xo?Y-_ecpzD5rG^<GSt{7!Vx67#P;JtV(<zbg@jq5_d2"
    "`R#JTB<GYPPLkh#mqI!5$e-iAZ5S{4wVkOXN%9+Z){ASMJ_*m!r$lwh@As15-E%FRxN4;=(uPRM?{LX)`jB@bd7>Q;LsID)5y;+F"
    "lHVVni3OTMF|_oAD7qrqmD=ChmmNRF@%Hjt(QWigaj%Jnp4Usofu^oZmHfJ|j6hj-U1V%hvNKvk#BCR*NPZVee!F-`)+N6?6gpZz"
    "e=45MGw`;XiXKyJ$f%>_u>Wb1Dfz87zz^?B!7P=Gt(f>w4BZ~Wz1s49hM2jy-jg4{@w>#8R!&B4=st)UlHZGs0?Cu-Ea;sRw_k;E"
    "(es9=(Elg8NPayeyWiAq+6!@Vn3~yP*TtxoYQ|J3cvIcP9YZ)Rw|QV$Rw%lj*Yly@S8?E(fo#dIQ(}qeFZmtZ=elU|<)K)7*9L>+"
    "H+j7svn0RaQ==GcTQ0n3eiTzBzpn!wiOWl5qvUsvn;o|+QaI?X<H<A!inqkU{sht`w+l8r6W?n4V;SuF=)>*jMlO2j82jO?2$B5u"
    "l>FY3JuPYx$QH@%$!0E9wZV*F3iE<_^k*El^-QdJ>q^P;G&ZOmaja!WeaUZo$*)bfN-_4S3cpGd9d}fS&kM|){#(b7Gp)?;m&$Y>"
    "73q^pMUdn-VVM*Aeh8wq<hQ5kvPkt^Ew)I0ZIk_JTC8NS?CYCuT5d$wVUgsw@vn~bnp+_vB){VVD?~ek4mZi~=(~2DaaHr6(oEyw"
    "UfekvN^yK5Q=?QgTb;_(cJ}0~&Sr38HYcz6@nbJNquc5^5_(w-8E@rZd2c`Uwx!>83!PKUtc<dw&IeEan3l;~H&410S*uhTqoDDh"
    "dVX6O&M_A|GG}ObP%9FLM-N2@dkb?4EZiSw!|>najEK)=fqd@gP2;N6!tzx(JG)VLbQE*^bm$tyvf)t{#z=22TvW0DWIU$jpGCYz"
    "L!a?2cyTC>UbnP-l6|e%;m@>TA(YrQ<mc(R4EfQI8LAX=6W!_5#)+;!SqPi#!P4!nJn%8nAorbExH*=Z1#;%*+cP3Vg}>pgSnGdB"
    "G(BYEEhc7NbVPkQ37d1F)V$zN$!rDlP8nJGha<U49erkav1a~9LC2FK<&KhZ*Oe?-W8vl%D}TxTn|%HM_ivq;*Wy53J4*Da{C+Qh"
    "Gc6ril<-V!T>U`osE_z`*_&?~P~^k$&XV7<H=>tsk$C%^hBY(ah>KQNmd(v%O|J-=%)2UVmN;@UO2g-VE_}(#V6$Yn^SfXsZOOs*"
    "gqB&VV)1#d0XI35zD`omIz!3-!cK^T@;&PJ`%&tvVOW@&7luNSzbk~dwahFEGBa?qCxfJSFj!m(-JC!pyHc^zBbKYn0=QhmNb|W)"
    "91&r-CS4W!j)fxcc_1MMeU<jqe<p6_srfVKU$LN#ntsz1ypJ^T@w|boJMIjqc3&i%)05Tji|97OK-)<c&NqE4#*YZX<@^<qKdMmZ"
    "`rBY_6u{@%b~GMgVav=Yy!Mxgb{#&5rLM6wIqN`^+(Zs)(hwtT+5L^*X<8hsIZ$ej<ITPVn*LAAo{mq&w>so`2LBu3i>bDWlHOYW"
    "TJ}Yl^OG3aD2%-!uI&3hki&ncGuh3hs^&WflkO1AhwX9P(3l9^<I0h7Y5dyRfhUb@X?r?>{+HCW=>JK4w?;)HUo+Yl<-$wy8#hPt"
    "n``C20jUgr?#p1eQW4X{NM@=ND_nx;<(f?7w2LCZcBMFSPC;>=A4OhDCdV3C-&)IvfI7rVet)m)NX6B1kz8bA$k}q?_gagth1{7p"
    "?fC0-5b@<^Mo#e}=3OWiafzhPQ*mNtD%DQfb7-a9zm2omIm@4`J@l-Q{0`c4L0IJ6&YfW;`HU@vTP<8}YvysX9V4%M5%^msriY%~"
    "_|M8UsWuH13{&g*x;&iP-nOi}t)WF`B;(5-iNS6bly@xzM%hqZ&WK~Nx$Jo@pSww1m43fcDW2_;eH|Hvb-$KXHDd{Wn8nRb-t2g!"
    "BIZOq;kQ4Fe$waMAKQZT^KpcgYv?EYy5OiE+n$Ed)21Q%(YgFG+aFus6v~@;u=k}>Uc<uFHXf)(xYEMg$nOPjh1;fB)+K5wJYvtw"
    "G!<<dy%L_|Z;Azy-+9MOIJH$W^I{U(&@j0>0|-<qa5`zEm(r2L4mzgK^P)+QkHV+d2{G%Ik{*|oEV^K!(Pb;w3vJkXHG~z5^gK>@"
    "A@YB>qo*#FgTaAhv~Xnf)~8~9^9SP4hhw7uC2ziIvBj6<PI1&)_*VQ^S}3|T*03x3h3F|~j`y5QO8<(W!?epH;e-Pz)in&-=t6nz"
    "40MuV(|QdbH|21*zK-aSA~A4|fw+z;)_Q#vznPS5^FAh8jkRLC)}OL64aXE}GMyiYy&FR4mT6|cr<vS7UObVrCra+7e|97=(NrQ-"
    "?y+o91@LEeBVos!SQH@ldb>;F+{}C8+0#Jghw7{J_GPhXJXFou-ZzAMJ2hukDCihrqV$o02SeSNzOO(GJgcW?-7jKfM+4_4SV(*G"
    "N|X)@qWd40MEtA!qQPecn{xx0V{cE=FbiEPqA1<}LDU#fD!g3e9I5R{gN=#A2TT9F(iX?hDfAIqo`yTH@b6fL?M>i#e=W6J7mIIo"
    "lSd|nvsE+<G;?gGmOl5thz*mHc%2)@qK&Tn`8E*ap>)<7T&n8PV`>_lRnvQ099z4ZP*l1S*fWhCZ5`<7X3P0g2|T=@#{P1dxFCC*"
    "A7Lgoy+TZyXQtM49ZiZ7xheTg8Li^6-v_a(v1BvKiGfZ*6gVf-H~xa4*>Yhzt3cV_k2_Bt`Bc+LwVhfHD(lcJurBpiJCf14LTvrd"
    "MAH5WF{H7Mx!+sZmt@aRFN28qWM+%Wo5y3r7!aGt)*XH<UY?4_a|dQG&*tL~+042Vz{YNRW=a2SeRN)APmptanw3>E?by57Lgi#L"
    "J{Rn0ZR^9UUo)Ba$&0OztgM#`eXn4w^u*VehLd;5h8}gbM5jlx`1F6mzM6%*w=8t?v*AW(e}1o-OM>*jmd0^aI+;=-454n!7#@ZC"
    "p5#jI-?#U(z}|<XfqvK#&zQPjMfzcRuE(^XdqzB~E@`ME``YXmKV~_Fk?~btvwtqxzJc8EN@48*Pa<C_+1k;<{EZ&0J?2W3r;*Qj"
    "??mDHSXS8U_%qJ|QdGEHc`oJ|{}sWK-`j^x49Jmt`+O3UZiMmtVF1f#e-#B7N#5(gJUbl?9(pnD`A0GKf5*iGxqE*-rDUCbA|V&8"
    "^f+nD#LFRU>#UdA>Z#Z;)Se7YD#5n{@&4YC?$?V&K;?ar7$8KQ3*LOw`a~5T9pl)&>b*#5a!;f*)S#&>7LJ?Tcrr7S$^9Zxf4L}z"
    "#XH~`p<z<K^mmyVRq7e7<NSskJbu=b>{%qTXBilLU3!)#pT+Y~>9=1W5|c+;d0-5rWQ>k`l|h^sa$g*pA3{pHnHM(lToie8@RF6x"
    "5?6vlq(?dUUObcB_#g16f25JMqnv5@%D{%T7eq+WU2&u+kluQIl{QW)7NLXHC}!Oh=JwK?r`S--U}Ai?aIVa9r{U}Z;eJNXj<PQz"
    "Y>9z2<1CmuzZ7SO2FcG}6svx@C+f&-JT51IHYN6K9AaVmsb~&+m5IhzOT`mqEO$pJ>GGQuhoCgZO|~OS@;hm-j?i5W7?NUXv^#-C"
    "eYHGUS0ujGgc_N&NbyH>vrzK0j?+IX7&9S><2hk$yX(fD8bLT6NaxOcm#SJU)ZD+Nrk3>LW;ZjHd)#Q$HVp@b<ZOs7YfdDv_nexY"
    "zm*BE>MAb8n3=e{Qna08X4pm@!@4ID*gus_{CKnEqsVJ$B=RRG+J6_sPGvH4BF~G*F3W_|X$2>q__1<;l8ezs=p!@F*DO-~>vCa*"
    "l4YjPV&WqcF&AaFHp`jU)I$CjdzueX^S9(Tc9;*AP6q1MOl09m>5Z18GW>{=f0ksU=#kBU`PI1plb&JiCBK_5h*LjXX*I>lj6da`"
    "+F;?=F=kRO+w-)yFNXX~JTrZG^gx~ssUNG`5*Me(b#XX70&S^rTuY~vNcP4Ti<24)zBeqynr&DmXZrc5T=dEi`sBt{sq?^cF+Rtg"
    "<3popJXpv5QL$7n$fAF_58i|QNj@4+r<V#G7if7tvIUhk30ONw?{Pc<e+Pd~o-i<^GM7``b9sBb8Xg`g{QR9aX&FvDZZBs?8&A4V"
    "bEA`+k<_3P5x6!MuXsI?^X2=os<?josi599vHyUH8j|6fBb`}tCW#)84J0e8F=zf)p*m({%O(fT$ecN3kT*)-a`AIb#KFr-vJWb0"
    "otj9Sb5;tz+Ohva2qBJoPJAj7XU;e<Bq)`7wn5Zs;>hi#MWS<``{Ke@X$8-E^G$c8C#l;$4(C@NM7H^^7?`7>Lt>G*>*j%GdM1j="
    ")p_D|QMeb|lN}~~{|7f}q-RtqU1kUSbvev!8OpHZMPkf!183Ixv3g>;nC`3O)~S7>-v}$STL)oTE;Db4njur}iyjj~7%1~lbg7AH"
    "K`P#zxAH@fJ8g=M^w{=Z=$&FIUf|DflHVWqx?oC*U`*Tvq5b);`0zN8RT_PjR(>oNZU3j{*u7iAPV!sO(v}G!CNwz_ge7?3zxJM}"
    "e@f5(R0V&;gmY_*^v_pb2%Et{{Cw<!a2Q@77B^S0w_X588aUE(poKyAYA|l+M`4{%CW0Me`MI|<J8vhE7LbN-ghSQ-{jE<3MSC21"
    "ygr7B(lcD@r6sCMk@!}>OUWd@Lm(|(tT-lxFkNTMuCYm!*9&9NN)P%MtI61#&M!?|t7?RIFjrH9$=eu56I%<buehUbo<{u!N|bqa"
    "+!6`AIxU~=w~u0@yNVgHa_%+yDrU_z)AX8-g%>R}?<4c?&H&<msT5mtj99Ka5qma>r}oMGFzK9lma<f2o>Y+P8^HYePB=(k<1TmD"
    "v+^u9sp?W_bS67U!Ttwwe+*Xe>ul-a8e4d*b>v**U_531O|GS4R^M<&MJ3YcRyBGql%74_g&!AYqwkwd+GI7ue$?aejbGQx;`vA`"
    "uO?gZ3v*=hS_>m2zt^ifvge(OgA+2jKhKv@1=cEkdSJ(nw|Z7D2q(6N9dByuC`*jQ{qr-iFv7y7s}_z7v|;G60J=rw;<h;iyZUie"
    "O8l{0<OX=seoz$IZS;Iz9m|nhSu~7LG3jOimkz}fbljHk9XciqYeAB;kwFV|cpppPb9?|#93pZ3l*_ZOxilH6X01yK)o=RH4<|;o"
    "u`s;07xQnrQ|4r(KztC}SH-euQ3z|lDw%Co5!bX>INZA;R_v2!dB2JB!_rrsO5#n;2!;#^;^F(RV)Nfdj$Uz~^JgtxV&q)eQXv*M"
    "J|-@lS2AF;lHw7Gy#B|EVu}OJ&W12t?uXLnPsH&-N?Zd{iIh9)W<y7Esyz{3X516b+fNGbf4upooCg8iZ7utdSRsyYy(66JY3Nz{"
    "M6}%Q$*w7KPTq=Q=F<yeYMDI+It_UR9*E?ODmhIyuw-=(i*_1#AMjMDrW)9N%Ac9T%EWpnCB|L5gdxvL{|~{0)RVJ&pc;MF1JS2%"
    "2u@Zr2~SP5D)DE^87seRmb^VNu}oJY_S(nNc(*?jW&SPu!HtA_kzA^{AU3os5Yr0-Nea?e>9ppVc+*!+?df-fgXDL@Jv-dACOMNM"
    "nbO3Q$%PNZZlR}k6C0ddBKY4(3uP@|iiHD$`0u-m;?(WC!e6reT~+|CA1GPh&%zgN42-W3c^%5dE}K|xDBLJ%m%>%wH0q325>c7L"
    "FSo*Y?dycAGKQe-3H0c$rTg+C@vSbcohkD`5M?)#IIuU2@NM=q9G%3RtT3F8d2#WRhU+`i`C@gesy}R$y?Gmq`?@&3N;amMJ(<xi"
    "jp=%4?v~m!@JIraPpTQWp<Jx2rlOY7%-8?f$XqM^Xqk?UF&6&ok;;qMApX6qpl&@Q6$6}!>?6Nplg!aJ=fttsOT@oI!RbqZJgVcu"
    "QiI&Tzl5-<EDKNXx@@iQ%2ll`O$$uC{mGV_4Z=vwm0o3xlH~Rp%D$R8+}@v`x<%j<k;tR%!E~CJN}k4@R`asC);*g#tu^$Q?9P?>"
    "H|x_i;W=F9^6_#m-&Znvm4zO^n%THt$);ic?EE>C?As~^-Lh6G&dZT`5uq%a6HfIH_Go0Te{GE9y^YM4CJU?NjGA!FhNlez`5`oy"
    "y{Vy;XT?>i;jMCE9q3K-{!ujR7Q(UZvFy2?g-w4ylJcwZ&%Su-dpq!heF!54w;(dd#9zJS`TIM8=^q18E{nphG?zc*{+*)Gu*@-q"
    "yJ0Gh$hkeJg@w@C-uSHY<dwaVZe7d8x@B^=goR;T<jk+}a{rDm66xA2V%}~Or}mgw_e6Sak~lp{`v0$LPG%`sz2C^sI!D@m((=B{"
    "hwXhniy?m=7PZbOiTYhh{((fE9JdnFK}pUj`3y63)SUE0JWX}R&Nr2NHX1J0ccdi$u}GR&AR2r;BVOT+({XRc2L*Apc^suS3Q7;%"
    "6^hy#`qnQNb31smd2%NDx-}TE>Y^}pci@^@!x5D?`z;w&ved24oaH&>500cy!875SZ=gp=AgS|8L}Pm;cOU;LmJG78{f3S!4I^<Z"
    "R5O2Qp_tY@gs<`(j=g6hqGb@{PFgXK^WwX|%<K**726fDJP!=ui_yrY1|Ce=FFR6xNkk2}FD4ZP66de4(v%&~MR89xzGn)=t+r|!"
    "?{VNuH4{fqRHx{s7fm#eMgBoOofpeop^c<xo`uf$Ux`_Lg5>>P5=Xb(6@6PPsFxAI+o{gX?PcMa^u*n2eHGu;_#!rcj-f%IC)V;5"
    "&UmJgJ<0`5MGAFGBe=Q7nZjl@@!p(3$Dg!Z=~OJfRTs(c(@q-p{+z<-4-s;1JCQX!i76Rjw14h{;)To{+tYcO;9gZ7$7_fv(crK;"
    "j)J;YdLH#+?SM2gySOsrlJo=z6Br{swrcTb5!pmVOWC){OYP7~9uC;)Ih<$aP?uEfE9DG0Wk+glBer2Kw9yAM{Bsh%JI;wUrlsQi"
    "VFgp{)%0!Q#yq`|3EQM+lKxrkQWxzJcP9U4PtGk9duQ2`kQPo%w(Qk&CmIga($LPrtsa3C+C}0Vn#jNlI`+;?rN;L%!_Lg+Ws7W@"
    "?bdRlm7cXy*E-!6H-}nD8)IeJ181%-weW7SnPdH(Nht`xWn?B9AN@Fe&03`|Yn^z$I*d2d!?DYgd#yxAc5Eco^>2l;wuO&pEO;7h"
    ">AEhEotj({cZSj|Bd$uHHdlz><*wb-D~h77p=7Ly#r<*?lMnhczLgr?u6X!MN#BW~T<YHf?LIRdc7^fdz66=E)jV%hlXh=%+0sv5"
    "x4M=Rn-scv`LV2~3#Xe{XqM%}+q>S>Q%KM9{EPU!FqV@m3_Qto<$etny1_-_isp)#`Im``9VU8Rabw)!B=YxECw#DmcMBDi?=mud"
    "vLk<%X!&QCFP;ux#k$D{#jBG_&X1S+lt}tfE0x2YSs_ArJ4nZdB~QfOcP_N^Naez0EzxxxS!jMFT28ww?xbH5Ck}h_;}LIEk-<ba"
    "iQ{r>TO2#w7p>DY+)j8dk{<eCJ3f;QYicsE)@8B7;z*4E4X?ZV;+2q5CH113JX@5*AHAZv^5%v3X}p0?zgHvN<DICZR1&awtN5e8"
    "m5FOY89X$a%0|I-D10c6riRc;`oOg7CT7@bxpB<O%q~8pzcW*B&IjRG8N*xxxSD9>zC0sq=hmQ={WURY(L-_PW+3_A`YN^T{ZeFh"
    "Rr7Vj197Rf8pB>Co%~Gn4Ugu<Mjuvs7mM&6GFMgFv88Qw>A~c#YVcMRbq``><EtX-!d=mFt%7CA0odzYiItgkO<^o~XKh&9zy{k-"
    "F?`qGhd75+LR{0Zb@$-#rxdgsquFJ3Wze!3^jw#~Zkea%ta>WG)iW8H{CZqRl4mOAkD{3$?Z%?PNjyjn<MM75`8V_!HmB3W;KAEX"
    "=`5eBrA4xiAC|?j(;<o09erp#HH~2f?##dC#F0G-T>D$i{t*h?x2gz``S<2m2Zl~GGow<6@{Eax9aEWmSjTM1`<yiC^Zs+;Lvb({"
    "KPFLs-gyx(^RMB6g1(b9+?nmcbG4B^=fe2Ku?{`$>QdRy3;R?hE3TO+z2?ZIoJb}~Zv2fd<agEKD*0V;tr`;zk#y1|Vtf}$zp1IT"
    "o8U{AsoA`?W@F6LF{+uKZcX*f3n~-=gRQ(9W+f-wjVZrdus>kNKGT(nfz>!VB$J~r{26=ES|yv=t{5g6SpRD{-B!8q?<hT$)g$??"
    "L5Wx=bNIFs7R>8xnf@e@UV*t>*My-=j;qqa4V6Ow&Yw<QqX?-8W#VtKa5js(p946cm0o9CJjDZCcu>PYey<k%+B%WnG9&QYnZS{Y"
    "8lIh#dG>iOwMOPL^<OQUDwAos$d3uGu8fr)<bG{m7LHX>S)PE$w=6U_mI<fBad(mLN6w8Yk;UTCgDb*ztBHT4?v8gSaeoq57RkN7"
    "N?xx*!JpfW?6pzS{k5F$*(yeaD#)+1PsF2STR$a#nXJ4&U}f8FS6coZ!kMNzLT!sgr#v^joKso+pN_}rj!f<QQ0y+dBObiEF4*Ud"
    "-vMtf@73_2K^$G`+4E%lBjIk<pxyCW4E0yBe^e&R?!<6)|5Y)sqmn{z4I3BxaiT^BC#C+^B7v3jayZi`hLnugB5<^U`|fI%Yu*Th"
    "yAn_T&Ei=vD_h?i7}__6^-F_UnekXOjSRsp%FO5UCf3VbJpGWBKlLhl6`Hv}_M<rTAqI!7fh@Kf$#3dI`)f7nb?1h#S@}fFz8XlH"
    "o4!i9wy#BLM>TI$k40ZOYyA&8WAHK|qHAD3U&W#CUx<(E^ek`bK)*Fn@*XVcTbGDsor8GU<6rTUXMr#V+R)q_!1zgS%<N=gN}V`<"
    "a<%8)Y&-V7k0EWLAHT>R-f&3cA9o+#m8Qu1j-})eH?BF=K)Wh|K5ev&&v-7r)#TJnLN0{Rrfw=9I>+%r=}puANh~mjaqGGt9&7d7"
    "UYpKhTTklC>Me6wi$C@FEQ~`;OroDcg<@kGagDrCKXPH{UkOazrzR@UmP<AKNDepCv%eEHel}xyr=wqElk}{q_&SI3zX{Id$ey*{"
    "<H|~THYdMN;=j@hB74Acac7r;0n)Ri9QI<Bzme#N1{~!6o%N*-O|`!C{_0H2izaM6oe4Hak&<RX8}7<9KRsW*{eL4Wi1^zP1Oz5h"
    "a3);#E0s6j`!jq}HUk2)$=<DFW<x!i96j6a7K>YQZY!ntI=9t>WwR|Dy=Uf2j62zuAg=Y#B;jlTb5C2VRPUGvn+gpw7l%{O&4Z31"
    "A$WyF(s*Q<IM=}fM=Zp(vtx8bHGcNVC2md_YExX5+V1!)5+_$<NXIB1x`Z*kcPuwgWN}!2HeY&zryJtwu8_a`HJl|~T98vciGNR4"
    "Cu(Z~4+iSlQV@^fzg&E$=Av@dari?rXD<11+(Y`5x)ur>s&E<+fLCb(<3b$xXS$s4D<g=$?ZHNQW}|+6Dkgn@O^jM^;-r+>!-Ix9"
    "lPC&{#%ZJuzo9l9-)Q8^LM0=gN{_NqMf7|H%7ME@@4qFxt(9!<WW{N>mHFd6=(RV5Ml$~&nj!PpDR&HZsU#UfD7QLd{OO@k+uju("
    "VRyvP9o|Im^2V}DOQ}4w^Oibr!uP2NNzmYQpi~6DQgME$%yP$Kx%cc}(eRQJA6zw5pBTXT@C@!qCDlx1*Nhx4c8EuJ?5&8&Gw^%E"
    "U>bgXCFXi5NtnG+Z0cfV*5nA@kB#H|RlyAC@SpI|gb)&DCi@=~f9y4|dXJTH3;cM7nThAi#ichfy#J>fTjW`66|LgQxEKaU-xZ^b"
    "&&9g)fi!c}SE*5(H^N!^=J-=ZqD@OR^VYkP;$mXXts30!;ZJ_^H)8P;J=Qsn(l19drMrbihswlX?SrWP<dz7pUmylAwV_9B0PhaF"
    "^GkaRgV)EQi+03ww*%*1#<1HrkP2IwFTbWTw!S}o-lPzuH__>;I}KJwV_G8L>-SphKfMs&>XIb6TSkU4J3p0P31;%A_|vs#5}RYg"
    "xEkiqv+jCctw?9c1y5ctPiN{gJzaA`iI^kbfqOCo{_rD<H2nMd;CIH2snVY>-=SvK|LiE=@5c_cS>_EFY7I9t|D%p0@_E-aPlefG"
    "=<{9akYJ?eJU9FbwJds>MAGR?qGOwt;`KHKyM~2QZLJTxy^NeG52wak>7Og=p#0sBoV9NBI&DH7>&o~0qiK>TGwT6Y+KkuB-!U_!"
    "Yc(pHM^M8jk^cin<dho"
)
_POWER_MAP_SHAPE = (11, N_LOCATIONS, N_SECTORS)
_POWER_MAP_CACHE: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}


def _power_maps(reference: torch.Tensor) -> torch.Tensor:
    """Decode the immutable published map subset once per device and dtype."""

    key = (reference.device.type, reference.device.index, reference.dtype)
    cached = _POWER_MAP_CACHE.get(key)
    if cached is None:
        raw = zlib.decompress(base64.b85decode(_EMBEDDED_POWER_MAP_B85.encode()))
        array = np.frombuffer(raw, dtype="<f4").reshape(_POWER_MAP_SHAPE).copy()
        cached = torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
        _POWER_MAP_CACHE[key] = cached
    return cached


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Return RSRP and non-serving interference at the 16 raster locations."""

    original_shape = X.shape[:-1]
    flat = X.double().reshape(-1, DIM).clamp(0.0, 1.0)
    downtilts = 10.0 * flat[:, :N_SECTORS]
    transmit_power_dbm = 30.0 + 20.0 * flat[:, N_SECTORS:]
    maps = _power_maps(flat)

    # Match the released simulator: interpolate each sector between adjacent
    # integer-downtilt maps, then add that sector's configured transmit power.
    lower = torch.floor(downtilts).to(torch.long).clamp(0, 9)
    fraction = downtilts - lower.to(downtilts)
    sector_maps = []
    for sector in range(N_SECTORS):
        low = maps[lower[:, sector], :, sector]
        high = maps[lower[:, sector] + 1, :, sector]
        interpolated = low + fraction[:, sector, None] * (high - low)
        sector_maps.append(interpolated + transmit_power_dbm[:, sector, None])
    received_dbm = torch.stack(sector_maps, dim=-1)

    # The strongest sector serves each location.  Every other sector contributes
    # interference, which must be summed in watts and converted back to dBm.
    rsrp_dbm = received_dbm.amax(dim=-1)
    received_watt = torch.pow(10.0, received_dbm / 10.0 - 3.0)
    serving_watt = received_watt.amax(dim=-1)
    interference_watt = (
        received_watt.sum(dim=-1) - serving_watt
    ).clamp_min(MIN_INTERFERENCE_POWER_WATT)
    interference_dbm = 10.0 * torch.log10(interference_watt) + 30.0

    components = torch.cat((rsrp_dbm, interference_dbm), dim=-1)
    return components.reshape(*original_shape, 2 * N_LOCATIONS)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Apply soft versions of the released weak/over-coverage penalties."""

    rsrp = H[..., :N_LOCATIONS].clamp(-220.0, -20.0)
    interference = H[..., N_LOCATIONS:].clamp(-220.0, -20.0)

    weak_margin = WEAK_COVERAGE_THRESHOLD_DBM - rsrp
    weak_loss = torch.sigmoid(weak_margin / SOFT_GATE_SCALE_DB).mean(dim=-1)

    covered_gate = torch.sigmoid(-weak_margin / SOFT_GATE_SCALE_DB)
    over_margin = interference + OVER_COVERAGE_MARGIN_DB - rsrp
    over_loss = (
        covered_gate * torch.sigmoid(over_margin / SOFT_GATE_SCALE_DB)
    ).mean(dim=-1)
    return torch.stack((weak_loss, over_loss), dim=-1)


PROBLEM = BenchmarkProblem(
    name="Cellular coverage/capacity optimization (2 objectives, 30 dimensions)",
    slug="cellular_coverage_2obj_30d",
    dim=DIM,
    num_objectives=2,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 0.65, dtype=torch.double),
    num_components=2 * N_LOCATIONS,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
