<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Create Waiter</title>

    @include('layouts.Employee.LinkHeader')

    <style>
        #video {
            width: 100%;
            height: auto;
            background-color: #ddd;
        }

        #canvas {
            display: none;
        }
    </style>
</head>

<body>
    <div id="app">
        <div class="main-wrapper main-wrapper-1">
            <div class="navbar-bg"></div>

            @include('layouts.Employee.Header')

            <div class="main-sidebar sidebar-style-2">
                @include('layouts.Employee.Sidebar')
            </div>

            <!-- Main Content -->
            <div class="main-content">
                <section class="section">
                    <div class="col-10 col-md-6 col-lg-12">
                        <div class="card">
                            <form method="POST" action="{{ route('employee.waiter.store') }}" enctype="multipart/form-data">
                                @csrf
                                <div class="card-header">
                                    <h4>Create Waiter</h4>
                                </div>
                                <div class="card-body">
                                    <div class="form-group">
                                        <!-- ... (previous form fields remain unchanged) ... -->

                                        <div class="row">
                                            <div class="col">
                                                <label>Name</label>
                                                <input type="text" class="form-control" name="name" required="">
                                            </div>

                                            <div class="col">
                                                <label>Email</label>
                                                <input type="email" class="form-control" name="email" required="">
                                            </div>
                                        </div>
                                        <br>
                                        <div class="row">
                                            <div class="col-4">
                                                <label>Phone</label>
                                                <input type="tel" class="form-control" name="phone" required="">
                                            </div>

                                            <div class="col-2">
                                                <label>Age</label>
                                                <input type="number" class="form-control" name="age" required="">
                                            </div>
                                        </div>
                                        <br>
                                        <div class="row">
                                            <div class="col-3">
                                                <label>Gender</label>
                                                <select class="form-control select2" name="gender" required>
                                                    <option value="1">Male</option>
                                                    <option value="0">Female</option>
                                                </select>
                                            </div>

                                            <div class="col-3">
                                                <label>Status</label>
                                                <select class="form-control select2" name="status" required>
                                                    <option value="1">Active</option>
                                                    <option value="0">Not Active</option>
                                                </select>
                                            </div>
                                        </div>

                                        <br>
                                        <br>

                                        <div class="row">
                                            <div class="col-12">
                                                <button type="button" id="startCamera" class="btn btn-primary">Add Face</button>
                                                <video id="video" autoplay></video>
                                                <canvas id="canvas"></canvas>
                                                <input type="hidden" name="faceImage" id="faceImage">
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="card-footer text-center">
                                    <button class="btn btn-primary" type="submit">Create</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    </div>

    @include('layouts.Employee.LinkJS')

    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface"></script>

    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let context = canvas.getContext('2d');
        let model;

        document.getElementById('startCamera').addEventListener('click', async function() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
                video.srcObject = stream;
                model = await blazeface.load();
                detectFace();
            } catch (err) {
                console.error("Error accessing the camera", err);
            }
        });

        async function detectFace() {
            const prediction = await model.estimateFaces(video, false);

            if (prediction.length > 0) {
                // Face detected, capture the image
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                document.getElementById('faceImage').value = canvas.toDataURL('image/jpeg');
                video.srcObject.getTracks().forEach(track => track.stop());
                video.style.display = 'none';
                canvas.style.display = 'block';
            } else {
                // No face detected, try again
                requestAnimationFrame(detectFace);
            }
        }
    </script>
</body>

</html>
