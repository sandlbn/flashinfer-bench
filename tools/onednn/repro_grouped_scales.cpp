// Reproducer: oneDNN matmul weight scales set via the `groups` argument are silently
// wrong on Intel GPU. Ungrouped per-column scales are exact.
//
//   per-column scale (N values), uniform         OK
//   per-column scale (N values), varying         OK
//   N-grouped scale (NB values), uniform         ** WRONG ** (30/40 sampled elements)
//   N-grouped scale (NB values), varying         ** WRONG ** (30/40 sampled elements)
//
// Grouping along K is wrong too, and there by exactly a factor of two independent of
// M, N, K and the group extents.
//
// Note the failure is invisible to a one-element check: C[0,0] is among the elements that
// happen to be right. It is also invisible to a check using a uniform scale value. This
// samples 40 positions with varying scales for that reason.
//
// Observed identically on oneDNN 3.11.4 (oneAPI 2026.0) and a source build of 3.13.2,
// Intel Arc B580, Linux. Both report impl=jit:gemm:any and create the primitive
// descriptor without complaint.
//
// Build:
//   icpx -fsycl -O2 -std=c++17 -I$DNNLROOT/include -L$DNNLROOT/lib \
//        -Wl,-rpath,$DNNLROOT/lib -ldnnl repro_grouped_scales.cpp -o repro
//
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
using namespace dnnl; using memory_t=dnnl::memory;
static float dec(uint8_t b){ return b==0x38?1.f:(b==0xB8?-1.f:0.5f); }
static uint8_t enc(float v){ return v==1.f?0x38:(v==-1.f?0xB8:0x30); }
int main(){
  const int M=256,N=512,K=256,BN=128; const int NB=N/BN;
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  auto eng=sycl_interop::make_engine(q.get_device(),q.get_context());
  auto strm=sycl_interop::make_stream(eng,q);
  std::vector<uint8_t> hA(size_t(M)*K), hB(size_t(N)*K);
  for(size_t i=0;i<hA.size();++i) hA[i]=enc(i%3==0?1.f:(i%3==1?-1.f:0.5f));
  for(size_t i=0;i<hB.size();++i) hB[i]=enc(i%2==0?1.f:0.5f);
  auto *dA=sycl::malloc_device<uint8_t>(hA.size(),q); auto *dB=sycl::malloc_device<uint8_t>(hB.size(),q);
  auto *dS=sycl::malloc_device<float>(size_t(N),q); auto *dC=sycl::malloc_device<float>(size_t(M)*N,q);
  q.memcpy(dA,hA.data(),hA.size()).wait(); q.memcpy(dB,hB.data(),hB.size()).wait();

  auto go=[&](const char*name,bool grouped,bool vary){
    int cnt = grouped? NB : N;
    std::vector<float> hs(cnt);
    for(int i=0;i<cnt;i++) hs[i]= vary ? (0.5f+0.25f*float(i%4)) : 0.5f;
    q.memcpy(dS,hs.data(),cnt*4).wait(); q.memset(dC,0,size_t(M)*N*4).wait();
    memory_t::desc a_md({M,K},memory_t::data_type::f8_e4m3,memory_t::dims{K,1});
    memory_t::desc b_md({K,N},memory_t::data_type::f8_e4m3,memory_t::dims{1,K});
    memory_t::desc c_md({M,N},memory_t::data_type::f32,memory_t::dims{N,1});
    primitive_attr attr;
    if(grouped) attr.set_scales(DNNL_ARG_WEIGHTS,1<<1,{1,BN},memory_t::data_type::f32);
    else        attr.set_scales_mask(DNNL_ARG_WEIGHTS,1<<1);
    matmul::primitive_desc pd(eng,a_md,b_md,c_md,attr); matmul p(pd);
    p.execute(strm,{{DNNL_ARG_SRC,memory_t(a_md,eng,dA)},{DNNL_ARG_WEIGHTS,memory_t(b_md,eng,dB)},
      {DNNL_ARG_DST,memory_t(c_md,eng,dC)},
      {DNNL_ARG_ATTR_SCALES|DNNL_ARG_WEIGHTS,
        memory_t(memory_t::desc({cnt},memory_t::data_type::f32,memory_t::dims{1}),eng,dS)}});
    strm.wait();
    std::vector<float> hC(size_t(M)*N); q.memcpy(hC.data(),dC,hC.size()*4).wait();
    int bad=0;
    for(int t=0;t<40;t++){ int m=(t*23)%M,n=(t*97)%N; double ref=0;
      double sc = grouped? hs[n/BN] : hs[n];
      for(int k=0;k<K;k++) ref += dec(hA[size_t(m)*K+k])*dec(hB[size_t(n)*K+k])*sc;
      if(std::fabs(hC[size_t(m)*N+n]-ref)>1e-3*std::fabs(ref)+1e-3) bad++; }
    printf("  %-44s %s (%d/40 bad)\n",name,bad?"** WRONG **":"OK",bad);
  };
  go("per-column scale (N values), uniform",   false,false);
  go("per-column scale (N values), varying",   false,true );
  go("N-grouped scale (NB values), uniform",   true, false);
  go("N-grouped scale (NB values), varying",   true, true );
  return 0;
}
