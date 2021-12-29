load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive","git_repository")

new_local_repository(
    name = "opencv",
    path = "third_party/opencv",
    build_file = "third_party/opencv/opencv.BUILD",
)

http_archive(
    name = "eigen",
    build_file = "//:eigen.BUILD",
    sha256 = "3a66f9bfce85aff39bc255d5a341f87336ec6f5911e8d816dd4a3fdc500f8acf",
    url = "https://bitbucket.org/eigen/eigen/get/c5e90d9.tar.gz",
    strip_prefix="eigen-eigen-c5e90d9e764e"
)
