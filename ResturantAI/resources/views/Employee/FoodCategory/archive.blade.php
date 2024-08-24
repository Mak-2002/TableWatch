<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Food Category Archive Table</title>

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

                    <div class="row">
                        <div class="col-12">
                            <div class="card">
                                <div class="card-header">
                                    <h4>Food Category Archive Table</h4>
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
                                    <div class="table-responsive">
                                        <table class="table table-striped" id="table-1">
                                            <thead>
                                                <tr>
                                                    <th class="text-center">
                                                        #
                                                    </th>
                                                    <th>Name</th>
                                                    <th>Details</th>
                                                    <th>Food Note</th>
                                                    <th>Status</th>
                                                    <th>Image</th>
                                                    <th>Price</th>
                                                    <th>Note</th>
                                                    <th>Category Name</th>
                                                    <th>Category Type</th>
                                                    <th>Created By</th>
                                                    <th>Created Date</th>
                                                    <th>Last Update Date</th>
                                                    <th>Deleted Date</th>
                                                    <th>Action</th>
                                                </tr>
                                            </thead>

                                            <tbody>
                                                @foreach ($categoryFoods as $categoryFood)
                                                    <tr>
                                                        <td>
                                                            {{ $categoryFood->id }}
                                                        </td>
                                                        <td>{{ $categoryFood->food->name }}</td>
                                                        <td>

                                                            {{ $categoryFood->food->details }}
                                                        </td>
                                                        <td>
                                                            {{ $categoryFood->food->note }}
                                                        </td>

                                                        <td>
                                                            @if ($categoryFood->food->status == 1)
                                                                <div class="badge badge-success">Active</div>
                                                            @else
                                                                <div class="badge badge-danger">Not Active</div>
                                                            @endif

                                                        </td>

                                                        <td><img src="{{ asset('image/' . $categoryFood->food->img) }}"
                                                                style="width: 100px; height: 100px;"></td>

                                                        <td>{{ $categoryFood->price }}</td>
                                                        <td>{{ $categoryFood->note }}</td>

                                                        <td>{{ $categoryFood->category->name }}</td>
                                                        <td>
                                                            @if ($categoryFood->category->type == 0)
                                                                <div class="badge badge-success">Meal</div>
                                                            @elseif($categoryFood->category->type == 1)
                                                                <div class="badge badge-success">Sandwich</div>
                                                            @endif

                                                        </td>


                                                        <td>{{ $categoryFood->food->employee->name }}</td>
                                                        <td>
                                                            {{ $categoryFood->created_at }}
                                                        </td>
                                                        <td>
                                                            {{ $categoryFood->updated_at }}
                                                        </td>

                                                        <td>
                                                            {{ $categoryFood->deleted_at }}
                                                        </td>
                                                        <td>
                                                            <div class="btn-group">
                                                                <button type="button"
                                                                    class="btn btn-secondary dropdown-toggle"
                                                                    data-toggle="dropdown" aria-haspopup="true"
                                                                    aria-expanded="false">
                                                                    <span class="visually-hidden">Detail</span>
                                                                </button>
                                                                <div class="dropdown-menu">
                                                                    <a class="dropdown-item"
                                                                        href="{{ route('employee.food.category.restore', $categoryFood->id) }}"
                                                                        style="size: 20px;">Restore</a>
                                                                    <form
                                                                        action="{{ route('employee.food.category.forceDelete', $categoryFood->id) }}"
                                                                        method="POST">
                                                                        @csrf
                                                                        @method('delete')
                                                                        <button class="dropdown-item" type="submit"
                                                                            style="color: red">Delete</button>
                                                                    </form>

                                                                </div>
                                                            </div>
                                                        </td>

                                                    </tr>
                                                @endforeach

                                            </tbody>

                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                </section>
            </div>

        </div>
    </div>

    @include('layouts.Employee.LinkJS')
</body>

</html>
