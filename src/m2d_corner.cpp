#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <pybind11/numpy.h>

#ifdef _OPENMP
#include <omp.h>
#endif

constexpr double PI = 3.14159265358979323846;

struct Vec2 {
    double x;
    double y;
};

struct Segment {
    Vec2 a;
    Vec2 b;
};

struct AngleSegment {
    double a0;
    double d0;
    double a1;
    double d1;
};

struct LineSeg {
    double x1;
    double y1;
    double x2;
    double y2;
};

inline Vec2 operator-(const Vec2& a, const Vec2& b)
{
    return {a.x - b.x, a.y - b.y};
}

inline Vec2 operator+(const Vec2& a, const Vec2& b)
{
    return {a.x + b.x, a.y + b.y};
}

inline Vec2 operator*(const Vec2& a, double s)
{
    return {a.x * s, a.y * s};
}

inline double norm(const Vec2& v)
{
    return std::hypot(v.x, v.y);
}

double visibility_area(
    const std::vector<Segment>& inputSegments,
    const Vec2& observer,
    double maxDistance = 100.0,
    int numSamples = 120,
    double maxSubd = 5.0)
{
    //------------------------------------------------------------------
    // FILTER + SHIFT TO OBSERVER COORDS
    //------------------------------------------------------------------

    std::vector<Segment> validSegments;
    validSegments.reserve(inputSegments.size());

    for (const auto& seg : inputSegments)
    {
        Vec2 p0 = seg.a - observer;
        Vec2 p1 = seg.b - observer;

        double d0 = norm(p0);
        double d1 = norm(p1);

        if (d0 <= maxDistance || d1 <= maxDistance)
        {
            validSegments.push_back({p0, p1});
        }
    }

    //------------------------------------------------------------------
    // SUBDIVIDE LONG SEGMENTS
    //------------------------------------------------------------------

    std::vector<Segment> subdivided;
    subdivided.reserve(validSegments.size() * 2);

    for (const auto& seg : validSegments)
    {
        Vec2 diff = seg.b - seg.a;
        double L = norm(diff);

        int n = std::max(1, (int)std::ceil(L / maxSubd));

        for (int i = 0; i < n; ++i)
        {
            double t0 = (double)i / n;
            double t1 = (double)(i + 1) / n;

            subdivided.push_back({
                seg.a + diff * t0,
                seg.a + diff * t1
            });
        }
    }

    //------------------------------------------------------------------
    // ANGLE REPRESENTATION
    //------------------------------------------------------------------

    std::vector<AngleSegment> angleSegs;
    angleSegs.reserve(subdivided.size());

    for (const auto& seg : subdivided)
    {
        double a0 = std::atan2(seg.a.y, seg.a.x);
        double a1 = std::atan2(seg.b.y, seg.b.x);

        double d0 = norm(seg.a);
        double d1 = norm(seg.b);

        double diff = std::fmod(a1 - a0 + 2.0 * PI, 2.0 * PI);

        bool swap = diff > PI;

        if (swap)
        {
            std::swap(a0, a1);
            std::swap(d0, d1);
        }

        angleSegs.push_back({a0, d0, a1, d1});
    }

    //------------------------------------------------------------------
    // SORT
    //------------------------------------------------------------------

    std::stable_sort(
        angleSegs.begin(),
        angleSegs.end(),
        [](const AngleSegment& A, const AngleSegment& B)
        {
            double aa = std::fmod(A.a0 + 2.0 * PI, 2.0 * PI);
            double bb = std::fmod(B.a0 + 2.0 * PI, 2.0 * PI);
            return aa < bb;
        });

    //------------------------------------------------------------------
    // SPLIT WRAPAROUND SEGMENTS
    //------------------------------------------------------------------

    std::vector<LineSeg> lines;
    lines.reserve(angleSegs.size() * 2 + 1);

    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();

    for (const auto& s : angleSegs)
    {
        bool wrap =
            (s.a0 > s.a1) &&
            (s.a0 * s.a1 < 0.0) &&
            (std::abs(s.a0 - s.a1) >= PI);

        if (!wrap)
        {
            lines.push_back({
                s.a0, s.d0,
                s.a1, s.d1
            });

            xmin = std::min(xmin, std::min(s.a0, s.a1));
            xmax = std::max(xmax, std::max(s.a0, s.a1));
        }
        else
        {
            lines.push_back({
                s.a0, s.d0,
                PI, s.d1
            });

            lines.push_back({
                -PI, s.d0,
                s.a1, s.d1
            });

            xmin = -PI;
            xmax = PI;
        }
    }

    //------------------------------------------------------------------
    // BACKGROUND HORIZON
    //------------------------------------------------------------------

    lines.push_back({
        xmin,
        maxDistance,
        xmax,
        maxDistance
    });

    //------------------------------------------------------------------
    // SAMPLE ANGLES
    //------------------------------------------------------------------

    std::vector<double> xs(numSamples);

    for (int i = 0; i < numSamples; ++i)
    {
        xs[i] =
            xmin +
            (xmax - xmin) *
            ((double)i / (numSamples - 1));
    }

    //------------------------------------------------------------------
    // OPENMP LOWER ENVELOPE
    //------------------------------------------------------------------

    int nThreads = 1;

#ifdef _OPENMP
    nThreads = omp_get_max_threads();
#endif

    std::vector<std::vector<double>> localYs(
        nThreads,
        std::vector<double>(
            numSamples,
            std::numeric_limits<double>::infinity()));

#pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif

        auto& ys = localYs[tid];

#pragma omp for schedule(static)
        for (long long s = 0;
             s < (long long)lines.size();
             ++s)
        {
            const auto& seg = lines[s];

            double lo = std::min(seg.x1, seg.x2);
            double hi = std::max(seg.x1, seg.x2);

            auto it0 =
                std::lower_bound(xs.begin(), xs.end(), lo);

            auto it1 =
                std::upper_bound(xs.begin(), xs.end(), hi);

            int i0 = (int)(it0 - xs.begin());
            int i1 = (int)(it1 - xs.begin());

            double dx = seg.x2 - seg.x1;

            if (std::abs(dx) < 1e-12)
            {
                double y =
                    std::min(seg.y1, seg.y2);

                for (int i = i0; i < i1; ++i)
                    ys[i] = std::min(ys[i], y);
            }
            else
            {
                double slope =
                    (seg.y2 - seg.y1) / dx;

                for (int i = i0; i < i1; ++i)
                {
                    double y =
                        seg.y1 +
                        slope *
                        (xs[i] - seg.x1);

                    ys[i] = std::min(ys[i], y);
                }
            }
        }
    }

    //------------------------------------------------------------------
    // REDUCE THREAD-LOCAL MINIMA
    //------------------------------------------------------------------

    std::vector<double> ys(
        numSamples,
        std::numeric_limits<double>::infinity());

    for (int t = 0; t < nThreads; ++t)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            ys[i] =
                std::min(ys[i],
                         localYs[t][i]);
        }
    }

    //------------------------------------------------------------------
    // AREA
    //------------------------------------------------------------------

    double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < numSamples; ++i)
    {
        double y = std::max(0.0, ys[i]);
        sum += y * y;
    }

    return PI * (sum / numSamples);
}





/*
def visibility_polygon(segments, observer, max_distance = 100, num_samples = 120):
    start = time.time()
    segments = np.array(segments, dtype=float) # shape (N, 2, 2)
    segments_diffs = segments-observer
    segments_dists = np.hypot(segments_diffs[:,:,0],segments_diffs[:,:,1])
    segment_mask = (segments_dists[:,0] <= max_distance) | (segments_dists[:,1] <= max_distance)
    valid_segments = segments_diffs[segment_mask]
    valid_dists = segments_dists[segment_mask]
    
    ### break up long segments
    norms_curves = np.linalg.norm(valid_segments[:,1,:]-valid_segments[:,0,:],axis=1)
    max_subd = 5
    subdivision_mask = (norms_curves >= max_subd)
    long_segments = valid_segments[subdivision_mask]
    short_segments = valid_segments[~subdivision_mask]
    long_norms = norms_curves[subdivision_mask]
    short_dists = valid_dists[~subdivision_mask]
    n_segs  = np.ceil(long_norms / max_subd).astype(int)
    start = long_segments[:, 0, :]
    end   = long_segments[:, 1, :]
    vec   = end - start
    t = np.concatenate([np.linspace(0, 1, ns+1)[:-1] for ns in n_segs])
    t_next = np.concatenate([np.linspace(0, 1, ns+1)[1:] for ns in n_segs])
    start_rep = np.repeat(start, n_segs, axis=0)
    vec_rep   = np.repeat(vec, n_segs, axis=0)
    seg_start = start_rep + vec_rep * t[:, None]
    seg_end   = start_rep + vec_rep * t_next[:, None]
    long_subdivided = np.stack([seg_start, seg_end], axis=1)
    long_dists = np.hypot(long_subdivided[:,:,0],long_subdivided[:,:,1])
    valid_segments = np.vstack([short_segments, long_subdivided])
    valid_dists = np.vstack([short_dists, long_dists])
    ###
    
    segments_dirs = valid_segments/valid_dists[:,:,None]
    segments_angles = np.arctan2(segments_dirs[:,:,1], segments_dirs[:,:,0])
    
    #each row in segments_angles goes col1 -> col2: CCW
    angle_diff = (segments_angles[:,1] - segments_angles[:,0]) % (2*np.pi)
    swap_mask = angle_diff < 0
    swap_mask |= angle_diff > np.pi
    out = segments_angles.copy()
    out_d = valid_dists.copy()
    out[swap_mask, 0] = segments_angles[:,1][swap_mask]
    out[swap_mask, 1] = segments_angles[:,0][swap_mask]
    #adjust distances array accordingly
    out_d[swap_mask,0] = valid_dists[:,1][swap_mask]
    out_d[swap_mask,1] = valid_dists[:,0][swap_mask]
    #sort by angle CCW
    idx = np.argsort((out[:, 0] - 0.0) % (2*np.pi), kind='mergesort')
    out_dists = out_d[idx]
    out_angles = out[idx]

    ### if difference in a row >= π and (+) to (-) -> split and flip direction
    out_flip_mask = (out_angles[:,0] > out_angles[:,1]) & (out_angles[:,0]/out_angles[:,1] < 0) & (np.abs(out_angles[:,0]-out_angles[:,1])>=np.pi)
    ordered_angles = out_angles[~out_flip_mask]
    flipped_angles = out_angles[out_flip_mask]
    ordered_dists = out_dists[~out_flip_mask]
    flipped_dists = out_dists[out_flip_mask]
    n = flipped_angles.shape[0]
    flipped_1 = np.column_stack([flipped_angles[:, 0], np.pi * np.ones(n)])
    flipped_2 = np.column_stack([-np.pi * np.ones(n), flipped_angles[:, 1]])
    flipped_angles = np.vstack([flipped_1, flipped_2])
    flipped_d1 = np.column_stack([flipped_dists[:, 0], flipped_dists[:,1]])
    flipped_d2 = np.column_stack([flipped_dists[:, 0], flipped_dists[:,1]])
    flipped_dists = np.vstack([flipped_d1, flipped_d2])
    out_angles = np.vstack([ordered_angles,flipped_angles])
    out_dists = np.vstack([ordered_dists,flipped_dists])
    ###
    
    ##calculate area
    #project onto a line 0 -> 2π
    #intersection of intervals -> select lower bound
    tri_points1 = np.column_stack((out_angles[:,0],out_dists[:,0]))
    tri_points2 = np.column_stack((out_angles[:,1],out_dists[:,1]))
    tri_points = np.column_stack((tri_points1,tri_points2))
    tri_points = np.concatenate((tri_points,np.array([[out_angles.min(), max_distance, out_angles.max(), max_distance]])))

    segments = list(zip(tri_points[:,:2], tri_points[:,2:]))
    
    xs_all = [x for seg in segments for (x,_) in seg]
    xmin, xmax = min(xs_all), max(xs_all)
    xs = np.linspace(xmin, xmax, num_samples)
    ys = np.full_like(xs, np.inf, dtype=float)

    for (x1, y1), (x2, y2) in segments:
        mask = (xs >= min(x1,x2)) & (xs <= max(x1,x2))
        if x2 != x1:
            slope = (y2-y1)/(x2-x1)
            ys_seg = y1 + slope*(xs[mask]-x1)
        else:
            ys_seg = np.full(np.sum(mask), min(y1,y2))
        ys[mask] = np.minimum(ys[mask], ys_seg)
    ys_clipped = np.clip(ys, 0, None)
    #area = np.trapz(ys_clipped, xs)
    area = (ys_clipped**2).mean()*np.pi

    end = time.time()
    #print(end-start)
    return area
*/