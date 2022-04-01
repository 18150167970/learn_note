

```
ffmpeg -i "https://muse-skyriver.platoedu.net/live/6218845b2e10c5698eb10742.m3u8?nonce=1646018652&token=2a172a6506c8459aa6083a6bd36b725b" -vcodec copy -f segment -segment_list out.list -segment_time 900 -segment_atclocktime 1 -strftime 1 "video%Y-%m-%d_%H-%M-%S_1.mp4"
```

视频流地址：https://muse-skyriver.platoedu.net/live/6218845b2e10c5698eb10742.m3u8?nonce=1646018652&token=2a172a6506c8459aa6083a6bd36b725b 

segment_time: 分割视频时长，单位秒

segment_list：生成的视频列表

-f segment：模式 分割



# 参考文献 #