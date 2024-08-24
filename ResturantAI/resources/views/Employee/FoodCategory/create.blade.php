<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Create Food Category</title>

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
                            <form method="POST" action="{{ route('employee.food.category.store') }}"
                                enctype="multipart/form-data">
                                @csrf
                                <div class="card-header">
                                    <h4>Create Food Category</h4>
                                </div>

                                  {{-- message Section --}}

                                  @if (session('success_message'))
                                  <div class="alert alert-success">
                                      <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                                          <span aria-hidden="true">&times;</span>
                                      </button>
                                      {{ session('success_message') }}
                                  </div>
                              @endif

                              @if (session('error_message'))
                                  <div class="alert alert-danger">
                                      <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                                          <span aria-hidden="true">&times;</span>
                                      </button>
                                      {{ session('error_message') }}
                                  </div>
                              @endif
                              {{-- end  message Section --}}
                              
                                <div class="card-body">
                                    <div class="form-group">
                                        <div class="row">
                                            <div class="col-3">
                                                <label>Food</label>
                                                <select class="form-control select2" name="foodID" required>
                                                    @foreach ($foods as $food)
                                                        <option value="{{ $food->id }}">
                                                            {{ $food->name }} - {{ $food->details }}
                                                        </option>
                                                    @endforeach
                                                </select>
                                            </div>

                                            <div class="col-3">
                                                <label>Category</label>
                                                <select class="form-control select2" name="categoryID" required>
                                                    @foreach ($categories as $category)
                                                        <option value="{{ $category->id }}">
                                                            {{ $category->name }} - @if ($category->type == 0)
                                                                <div>Meal</div>
                                                            @elseif($category->type == 1)
                                                                <div>Sandwich</div>
                                                            @endif
                                                        </option>
                                                    @endforeach
                                                </select>
                                            </div>
                                        </div>

                                        <br>

                                        <div class="row">

                                            <div class="col-3">
                                                <label>Price</label>
                                                <input type="number" class="form-control" name="price"
                                                    required="">
                                            </div>

                                            <div class="col-3">
                                                <label>Note</label>
                                                <input type="text" class="form-control" name="note"
                                                    required="">
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
