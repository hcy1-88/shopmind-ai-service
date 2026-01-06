本地开发时，推荐以服务器运行，避免 File 协议跨域问题：

```shell
cd html;
python -m http.server 8000
```

在浏览器打开 http://localhost:8000/rag-manager.html 即可运行。

如果是部署，应该部署到 nginx
