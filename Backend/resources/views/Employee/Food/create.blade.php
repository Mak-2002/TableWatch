<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Create Food</title>

    @include('layouts.Employee.LinkHeader')

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
                            <form method="POST"
                                action="{{ route('employee.food.store') }}"
                                enctype="multipart/form-data">
                                @csrf
                                <div class="card-header">
                                    <h4>Create Food</h4>
                                </div>
                                <div class="card-body">
                                    <div class="form-group">
                                        <div class="row">
                                            <div class="col">
                                                <label>Name</label>
                                                <input type="text" class="form-control" name="name"
                                                     required="">
                                            </div>

                                            <div class="col">
                                                <label>Details</label>
                                                <input type="text" class="form-control" name="details"
                                                     required="">
                                            </div>
                                        </div>

                                        <br>

                                        <div class="row">

                                            <div class="col-4">
                                                <label>Note</label>
                                                <input type="text" class="form-control" name="note" required=""
                                                    >
                                            </div>

                                            <div class="col-3">
                                                <label>Status</label>
                                                <select class="form-control select2" name="status" required>
                                                    <option value="1"
                                                    >Active
                                                    </option>
                                                    <option value="0"
                                                     >Not
                                                        Active</option>
                                                </select>
                                            </div>

                                        </div>

                                        <br>

 

                                        <div class="row">

                                            <div class="col-5">
                                                <label>Image</label>
                                                <input type="file" class="form-control" name="img">
                                            </div>

                                        </div>
                                        <br>

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
</body>

</html>
