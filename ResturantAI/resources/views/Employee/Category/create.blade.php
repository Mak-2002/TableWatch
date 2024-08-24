<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Create Category</title>

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
                                action="{{ route('employee.category.store') }}"
                                enctype="multipart/form-data">
                                @csrf
                                <div class="card-header">
                                    <h4>Create Category</h4>
                                </div>
                                <div class="card-body">
                                    <div class="form-group">
                                        <div class="row">
                                            <div class="col">
                                                <label>Name</label>
                                                <input type="text" class="form-control" name="name"
                                                     required="">
                                            </div>

                                            <div class="col-3">
                                                <label>Type</label>
                                                <select class="form-control select2" name="type" required>
                                                    <option value="0"
                                                    >Meal
                                                    </option>
                                                    <option value="1"
                                                     >Sandwich
                                                        </option>
                                                </select>
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
