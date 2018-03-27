import sys
import numpy as np
from dart_board import constants as c

import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.optimize as so




def get_theta_proj_degree(ra, dec, ra_b, dec_b):
    """ Return angular distance between two points

    Parameters
    ----------
    ra : float64
        Right ascension of first coordinate (degrees)
    dec : float64
        Declination of first coordinate (degrees)
    ra_b : float64
        Right ascension of second coordinate (degrees)
    dec_b : float64
        Declination of second coordinate (degrees)

    Returns
    -------
    theta : float64
        Angular distance (radians)
    """

    ra1 = c.deg_to_rad * ra
    dec1 = c.deg_to_rad * dec
    ra2 = c.deg_to_rad * ra_b
    dec2 = c.deg_to_rad * dec_b

    return np.sqrt((ra1-ra2)**2 * np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)

def get_dist_closest(ra, dec, coor):
    """ Returns the distance to the closest star formation history region
    Parameters
    ----------
    ra : float64 or ndarray
        (Individual or ndarray of) right ascensions (degrees)
    dec : float64 or ndarray
        (Individual or ndarray of) declinations (degrees)
    coor : ndarray
        Array of already loaded LMC or SMC region coordinates

    Returns
    -------
    dist : float
        Distance to closest star formation history region (degrees)
    """

    ra1 = c.deg_to_rad * ra
    dec1 = c.deg_to_rad * dec
    ra2 = c.deg_to_rad * coor["ra"]
    dec2 = c.deg_to_rad * coor["dec"]

    dist = np.sqrt((ra1-ra2)**2*np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)
    index = np.argmin(dist)

    return c.rad_to_deg * dist[index]



def get_plot_polar(age, sfh_function=None, fig_in=None, ax=None, gs=None,
        ra_dist=None, dec_dist=None, ra_dist_2=None, dec_dist_2=None,
        contour_CL='gaussian', dist_bins=25, dist2_bins=25,
        sfh_bins=30, sfh_levels=None, ra=None, dec=None,
        xcenter=None, ycenter=None, xwidth=None, ywidth=None, rot_angle=0.0,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=8, ygrid_density=5,
        color_map='Blues', color_bar=False, colorbar_pad=-20, colorbar_label_y=1.1,
        contour_alpha=1.0, title=None):
    """ return a plot of the star formation history of the SMC at a particular age.
    In this case, the plot should be curvelinear, instead of flattened.

    Parameters
    ----------
    age : float
        Star formation history age to calculate (Myr)
    sfh_function : function with arguments (ra, dec, time)
        Function that provides the star formation history for a position and time
    fig : matplotlib.figure (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    rect : int
        Subplot number
    gs : gridspec object (optional)
        If supplied, plot goes inside gridspec object provided
    ra_dist, dec_dist : array (optional)
        If supplied, plots contours around the distribution of these inputs
    dist_bins : int (optional)
        Number of bins for ra_dist-dec_dist contours
    sfh_bins : int (optional)
        Number of bins for star formation history contourf
    ra, dec : float (optional)
        If supplied, plot a red star at these coordinates (degrees)
    xcenter, ycenter : float (optional)
        If supplied, center the x,y-axis on these coordinates
    xwidth, ywidth : float (optional)
        If supplied, determines the scale of the plot
    rot_angle : float (optional)
        Rotation angle for polar plot
    xlabel, ylabel : string (optional)
        X-axis, y-axis label
    xgrid_density, ygrid_density : int (optional)
        Density of RA, Dec grid axes
    color_map : string (optional)
        One of the color map options from plt.cmap
    color_bar : bool (optional)
        Add a color bar to the plot
    title : string
        Add a title to the plot. Default is the age.

    Returns
    -------
    plt : matplotlib.pyplot plot
        Contour plot of the star formation history
    """

    import mpl_toolkits.axisartist.angle_helper as angle_helper
    from matplotlib.projections import PolarAxes
    from matplotlib.transforms import Affine2D
    from mpl_toolkits.axisartist import SubplotHost
    from mpl_toolkits.axisartist import GridHelperCurveLinear
    import matplotlib.gridspec as gridspec



    if (c.ra_min is None) or (c.ra_max is None) or (c.dec_min is None) or (c.dec_max is None):
        print("You must provide ra and dec bounds.")
        exit(-1)

    if sfh_function is None:
        print("You must provide a function that gives the star formation history.")
        exit(-1)

    # global sf_coor
    # global sf_sfh
    # global sf_dist
    #
    # global ra_min
    # global ra_max
    # global dec_min
    # global dec_max

    #
    # if c.sf_scheme is None:
    #     c.sf_scheme = "SMC"
    #
    # if (sf_coor is None) or (sf_sfh is None):
    #     load_sf_history(z=0.008)
    #
    # if c.sf_scheme == "SMC":
    #     if xcenter is None: xcenter=0.0
    #     if ycenter is None: ycenter=17.3
    #     if xwidth is None: xwidth=2.0
    #     if ywidth is None: ywidth=2.0
    #
    # if c.sf_scheme == "LMC":
    #     if xcenter is None: xcenter=0.0
    #     if ycenter is None: ycenter=21.0
    #     if xwidth is None: xwidth=5.0
    #     if ywidth is None: ywidth=5.0
    #
    # if c.sf_scheme == 'NGC4244':
    #     if xcenter is None: xcenter = 0.0
    #     if ycenter is None: ycenter = 127.8
    #     if xwidth is None: xwidth = 0.3
    #     if ywidth is None: ywidth = 0.1
    #
    # if c.sf_scheme == 'NGC660':
    #     if xcenter is None: xcenter = 0.0
    #     if ycenter is None: ycenter = 103.64
    #     if xwidth is None: xwidth = 0.08
    #     if ywidth is None: ywidth = 0.05


    def curvelinear_test2(fig, gs=None, xcenter=0.0, ycenter=17.3, xwidth=1.5, ywidth=1.5,
            rot_angle=0.0, xlabel=xlabel, ylabel=ylabel, xgrid_density=8, ygrid_density=5):
        """
        polar projection, but in a rectangular box.
        """

        tr = Affine2D().translate(0,90)
        tr += Affine2D().scale(np.pi/180., 1.)
        tr += PolarAxes.PolarTransform()
        # if c.sf_scheme == "SMC":
        #     rot_angle = 1.34
        # if c.sf_scheme == "LMC":
        #     rot_angle = 0.2
        # if c.sf_scheme == "NGC4244":
        #     rot_angle = 4.636
        # if c.sf_scheme == "NGC660":
        #     rot_angle = 1.1212

        tr += Affine2D().rotate(rot_angle)  # This rotates the grid

        extreme_finder = angle_helper.ExtremeFinderCycle(10, 60,
                                                        lon_cycle = 360,
                                                        lat_cycle = None,
                                                        lon_minmax = None,
                                                        lat_minmax = (-90, np.inf),
                                                        )

        grid_locator1 = angle_helper.LocatorHMS(xgrid_density) #changes theta gridline count
        tick_formatter1 = angle_helper.FormatterHMS()
        grid_locator2 = angle_helper.LocatorDMS(ygrid_density) #changes theta gridline count
        tick_formatter2 = angle_helper.FormatterDMS()


        grid_helper = GridHelperCurveLinear(tr,
                                            extreme_finder=extreme_finder,
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=tick_formatter2
                                            )

        # ax1 = SubplotHost(fig, rect, grid_helper=grid_helper)
        if gs is None:
            ax1 = SubplotHost(fig, 111, grid_helper=grid_helper)
        else:
            ax1 = SubplotHost(fig, gs, grid_helper=grid_helper)



        # make ticklabels of right and top axis visible.
        ax1.axis["right"].major_ticklabels.set_visible(False)
        ax1.axis["top"].major_ticklabels.set_visible(False)
        ax1.axis["bottom"].major_ticklabels.set_visible(True) #Turn off?

        # let right and bottom axis show ticklabels for 1st coordinate (angle)
        ax1.axis["right"].get_helper().nth_coord_ticks=0
        ax1.axis["bottom"].get_helper().nth_coord_ticks=0


        fig.add_subplot(ax1)

        grid_helper = ax1.get_grid_helper()

        # These move the grid
        ax1.set_xlim(xcenter-xwidth, xcenter+xwidth) # moves the origin left-right in ax1
        ax1.set_ylim(ycenter-ywidth, ycenter+ywidth) # moves the origin up-down


        if xlabel is not None: ax1.set_xlabel(xlabel)
        if ylabel is not None: ax1.set_ylabel(ylabel)
        ax1.grid(True, linestyle='-')
        #ax1.grid(linestyle='--', which='x') # either keyword applies to both
        #ax1.grid(linestyle=':', which='y')  # sets of gridlines


        return ax1,tr


    # User supplied input
    if fig_in is None:
        fig = plt.figure(1, figsize=(8, 6))
        fig.clf()
    else:
        fig = fig_in


    # tr.transform_point((x, 0)) is always (0,0)
            # => (theta, r) in but (r, theta) out...
    ax1, tr = curvelinear_test2(fig, gs=gs, xcenter=xcenter, ycenter=ycenter,
                    xwidth=xwidth, ywidth=ywidth, rot_angle=rot_angle,
                    xlabel=xlabel, ylabel=ylabel,
                    xgrid_density=xgrid_density, ygrid_density=ygrid_density)


    sfr = np.array([])


    # if c.sf_scheme == "SMC":
    #     levels = np.linspace(1.0e7, 1.0e9, 10)
    #     bins = 25
    # if c.sf_scheme == "LMC":
    #     levels = np.linspace(1.0e7, 2.0e8, 10)
    #     bins = 30
    # if c.sf_scheme == "NGC4244":
    #     levels = np.linspace(0.1, 25, 25)
    #     bins = 80
    # if c.sf_scheme == "NGC660":
    #     levels = np.linspace(2.0, 10.0, 20)
    #     bins = 150




    # CREATING OUR OWN, LARGER GRID FOR STAR FORMATION CONTOURS
    x_tmp = np.linspace(c.ra_min, c.ra_max, sfh_bins)
    y_tmp = np.linspace(c.dec_min, c.dec_max, sfh_bins)

    XX, YY = np.meshgrid(x_tmp, y_tmp)


    for i in np.arange(len(XX.flatten())):
        sfr = np.append(sfr, sfh_function(XX.flatten()[i], YY.flatten()[i], age))


    out_test = tr.transform(np.array([XX.flatten(), YY.flatten()]).T)




    # Plot star formation histories on adjusted coordinates
    # Plot color contours with linear spacing

    # To convert from Msun/Myr/rad^2 to Msun/yr/deg.^2
    unit_factor = 1.0/1.0e6 * (np.pi/180.0)**2

    if sfh_levels is None:
        sf_plot = plt.tricontourf(out_test[:,0], out_test[:,1], sfr*unit_factor, cmap=color_map,
                                  extend='max', alpha=contour_alpha, rasterized=True)
    else:
        sf_plot = plt.tricontourf(out_test[:,0], out_test[:,1], sfr*unit_factor, cmap=color_map, levels=sfh_levels,
                                  extend='max', alpha=contour_alpha, rasterized=True)

    if color_bar:
        sf_plot = plt.colorbar()
        ticks = np.linspace(1.0e7, 2.0e8, 10) * unit_factor
        sf_plot.set_ticks(ticks)
        ticks = np.round(ticks, decimals=3)
        sf_plot.set_ticklabels(ticks.astype(str))
        # sf_plot.set_label(r'$M_{\odot}$ yr$^{-1}$ deg.$^{-2}$', labelpad=-20, y=0.65)
        sf_plot.set_label(r'$\frac{M_{\odot}}{{\rm yr\ deg.}^2}$', rotation=0, labelpad=colorbar_pad, y=colorbar_label_y, fontsize=14)


    if title is None:
        plt.title(str(int(age)) + ' Myr')
    else:
        plt.title(title)


    # Plot the contours defining the distributions of ra_dist and dec_dist

    def plot_dist(ra_dist, dec_dist, color, dist_bins):

        # Need this function
        def find_confidence_interval(x, pdf, confidence_level):
            return pdf[pdf > x].sum() - confidence_level

        # Transform distribution
        if (sys.version_info > (3, 0)):
            coor_dist_polar = tr.transform(np.array([ra_dist, dec_dist]).T)  # Python 3 code
        else:
            coor_dist_polar = tr.transform(zip(ra_dist, dec_dist))  # Python 2 code


        # Create grid for binaries
        ra_width = c.ra_max - c.ra_min
        dec_width = c.dec_max - c.dec_min
        x_tmp = np.array([c.ra_min-ra_width*0.1, c.ra_max+ra_width*0.1])
        y_tmp = np.array([c.dec_min-dec_width*0.1, c.dec_max+dec_width*0.1])
        XX, YY = np.meshgrid(x_tmp, y_tmp)
        xy_tmp = tr.transform(np.array([XX.flatten(), YY.flatten()]).T)

        range_coor = [[np.min(xy_tmp[:,0]), np.max(xy_tmp[:,0])], [np.min(xy_tmp[:,1]), np.max(xy_tmp[:,1])]]
        if ra is None or dec is None:
            range_coor = [[xcenter-xwidth, xcenter+xwidth], [ycenter-ywidth, ycenter+ywidth]]
        else:
            ra_transformed, dec_transformed = tr.transform(np.array([ra, dec]))
            range_coor = [[ra_transformed-xwidth, ra_transformed+xwidth], [dec_transformed-ywidth, dec_transformed+ywidth]]

        # Create 2D histogram
        nbins_x = dist_bins
        nbins_y = dist_bins
        coor_dist_polar = coor_dist_polar[~np.isnan(coor_dist_polar[:,0])]  # Get rid of NaN's
        # H, xedges, yedges = np.histogram2d(coor_dist_polar[:,0], coor_dist_polar[:,1], bins=(nbins_x,nbins_y), normed=True)
        H, xedges, yedges = np.histogram2d(coor_dist_polar[:,0], coor_dist_polar[:,1], bins=(nbins_x,nbins_y), range=range_coor, normed=True)
        x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
        y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))
        pdf = (H*(x_bin_sizes*y_bin_sizes))

        # Find intervals
        if contour_CL == 'gaussian':
            one_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
            two_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
            three_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.995))
        elif contour_CL == 'quad':
            one_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.25))
            two_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.50))
            three_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.75))
        else:
            print("ERROR: You must provide an available option for position contour levels.")
            print("\t Options include `gaussian`: 68%, 95%, and 99.5% confidence levels.")
            print("\t Options include `quad`: 25%, 50%, and 75% confidence levels.")
        levels = [one_quad, two_quad, three_quad]
        X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
        Z = pdf.T

        # Plot contours
        # contour = plt.contour(X, Y, Z, levels=levels[::-1], origin="lower", colors=color, rasterized=True, linewidths=[0.5, 1., 1.5])
        # contour = plt.contour(X, Y, Z, levels=levels[::-1], origin="lower", colors=color, rasterized=True,
        #                       linewidths=1.0, linestyles=['dotted','dashed','solid'])
        contour = plt.contour(X, Y, Z, levels=[levels[0]], origin="lower", colors=color, rasterized=True,
                              linewidths=1.5, alpha=1.0)
        contour = plt.contour(X, Y, Z, levels=[levels[1]], origin="lower", colors=color, rasterized=True,
                              linewidths=1.5, alpha=0.5)
        contour = plt.contour(X, Y, Z, levels=[levels[2]], origin="lower", colors=color, rasterized=True,
                              linewidths=1.5, alpha=0.3)



    if ra_dist_2 is not None and dec_dist_2 is not None:
        contour = plot_dist(ra_dist, dec_dist, 'C1', dist_bins)
        contour = plot_dist(ra_dist_2, dec_dist_2, 'C2', dist2_bins)
    elif ra_dist is not None and dec_dist is not None:
        contour = plot_dist(ra_dist, dec_dist, 'k', dist_bins)



    # Plot a star at the coordinate position, if supplied
    if ra is not None and dec is not None:

        # If only a single ra and dec, or a list of points
        if isinstance(ra, np.ndarray):
            coor_pol = tr.transform(np.array([ra, dec]).T)
            sf_plot = plt.scatter(coor_pol[:,0], coor_pol[:,1], color='r', s=25, marker=".", zorder=10)
            # sf_plot = plt.scatter(coor_pol[:,0], coor_pol[:,1], color='r', s=80, zorder=10,
            #                       facecolors='none', edgecolors='r', linewidth=2)
        else:
            coor_pol1, coor_pol2 = tr.transform(np.array([np.array([ra, ra]), np.array([dec, dec])]).T)
            sf_plot = plt.scatter(coor_pol1[0], coor_pol1[1], color='r', s=50, marker="*", zorder=10)



    return sf_plot, ax1







def get_plot_polar_positions(dist_ra, dist_dec, fig_in=None, ax=None, gs=None,
        dist_bins=25, levels=None, ra=None, dec=None,
        circle_ra=None, circle_dec=None, circle_radius=None,
        xcenter=None, ycenter=None, xwidth=None, ywidth=None, rot_angle=0.0,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=8, ygrid_density=5,
        color_map='Blues', color_bar=False, contour_alpha=1.0, title=None):
    """ return a plot of the star formation history of the SMC at a particular age.
    In this case, the plot should be curvelinear, instead of flattened.

    Parameters
    ----------
    positions : ndarray
        Positions of a stellar distribution
    fig : matplotlib.figure (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    rect : int
        Subplot number
    gs : gridspec object (optional)
        If supplied, plot goes inside gridspec object provided
    ra_dist, dec_dist : array (optional)
        If supplied, plots contours around the distribution of these inputs
    dist_bins : int (optional)
        Number of bins for ra_dist-dec_dist contours
    ra, dec : float (optional)
        If supplied, plot a red star at these coordinates (degrees)
    xcenter, ycenter : float (optional)
        If supplied, center the x,y-axis on these coordinates
    xwidth, ywidth : float (optional)
        If supplied, determines the scale of the plot
    rot_angle : float (optional)
        Rotation angle for polar plot
    xlabel, ylabel : string (optional)
        X-axis, y-axis label
    xgrid_density, ygrid_density : int (optional)
        Density of RA, Dec grid axes
    color_map : string (optional)
        One of the color map options from plt.cmap
    color_bar : bool (optional)
        Add a color bar to the plot
    title : string
        Add a title to the plot. Default is the age.

    Returns
    -------
    plt : matplotlib.pyplot plot
        Contour plot of the star formation history
    """

    import mpl_toolkits.axisartist.angle_helper as angle_helper
    from matplotlib.projections import PolarAxes
    from matplotlib.transforms import Affine2D
    from mpl_toolkits.axisartist import SubplotHost
    from mpl_toolkits.axisartist import GridHelperCurveLinear
    import matplotlib.gridspec as gridspec



    if (c.ra_min is None) or (c.ra_max is None) or (c.dec_min is None) or (c.dec_max is None):
        print("You must provide ra and dec bounds.")
        exit(-1)


    def curvelinear_test2(fig, gs=None, xcenter=0.0, ycenter=17.3, xwidth=1.5, ywidth=1.5,
            rot_angle=0.0, xlabel=xlabel, ylabel=ylabel, xgrid_density=8, ygrid_density=5):
        """
        polar projection, but in a rectangular box.
        """

        tr = Affine2D().translate(0,90)
        tr += Affine2D().scale(np.pi/180., 1.)
        tr += PolarAxes.PolarTransform()

        tr += Affine2D().rotate(rot_angle)  # This rotates the grid

        extreme_finder = angle_helper.ExtremeFinderCycle(10, 60,
                                                        lon_cycle = 360,
                                                        lat_cycle = None,
                                                        lon_minmax = None,
                                                        lat_minmax = (-90, np.inf),
                                                        )

        grid_locator1 = angle_helper.LocatorHMS(xgrid_density) #changes theta gridline count
        tick_formatter1 = angle_helper.FormatterHMS()
        grid_locator2 = angle_helper.LocatorDMS(ygrid_density) #changes theta gridline count
        tick_formatter2 = angle_helper.FormatterDMS()


        grid_helper = GridHelperCurveLinear(tr,
                                            extreme_finder=extreme_finder,
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=tick_formatter2
                                            )

        # ax1 = SubplotHost(fig, rect, grid_helper=grid_helper)
        if gs is None:
            ax1 = SubplotHost(fig, 111, grid_helper=grid_helper)
        else:
            ax1 = SubplotHost(fig, gs, grid_helper=grid_helper)



        # make ticklabels of right and top axis visible.
        ax1.axis["right"].major_ticklabels.set_visible(False)
        ax1.axis["top"].major_ticklabels.set_visible(False)
        ax1.axis["bottom"].major_ticklabels.set_visible(True) #Turn off?

        # let right and bottom axis show ticklabels for 1st coordinate (angle)
        ax1.axis["right"].get_helper().nth_coord_ticks=0
        ax1.axis["bottom"].get_helper().nth_coord_ticks=0


        fig.add_subplot(ax1)

        grid_helper = ax1.get_grid_helper()

        # These move the grid
        ax1.set_xlim(xcenter-xwidth, xcenter+xwidth) # moves the origin left-right in ax1
        ax1.set_ylim(ycenter-ywidth, ycenter+ywidth) # moves the origin up-down


        if xlabel is not None: ax1.set_xlabel(xlabel)
        if ylabel is not None: ax1.set_ylabel(ylabel)
        ax1.grid(True, linestyle='-')


        return ax1,tr


    # User supplied input
    if fig_in is None:
        fig = plt.figure(1, figsize=(8, 6))
        fig.clf()
    else:
        fig = fig_in


    # tr.transform_point((x, 0)) is always (0,0)
            # => (theta, r) in but (r, theta) out...
    ax1, tr = curvelinear_test2(fig, gs=gs, xcenter=xcenter, ycenter=ycenter,
                    xwidth=xwidth, ywidth=ywidth, rot_angle=rot_angle,
                    xlabel=xlabel, ylabel=ylabel,
                    xgrid_density=xgrid_density, ygrid_density=ygrid_density)



    stellar_distribution = np.array([])

    # CREATING OUR OWN, LARGER GRID FOR STAR FORMATION CONTOURS
    x_tmp = np.linspace(c.ra_min, c.ra_max, dist_bins)
    y_tmp = np.linspace(c.dec_min, c.dec_max, dist_bins)

    XX, YY = np.meshgrid(x_tmp, y_tmp)

    distribution_range = [[c.ra_min, c.ra_max], [c.dec_min, c.dec_max]]
    H, xedges, yedges = np.histogram2d(dist_ra, dist_dec,
                                       bins=dist_bins, range=distribution_range,
                                       normed=True)
    H = H.T

    X_bounds, Y_bounds = np.meshgrid(xedges, yedges)

    stellar_distribution = H.flatten()

    #
    # # Here is where the distribution is plotted
    # for i in np.arange(len(X_bounds.flatten())):
    #
    #     stellar_distribution = np.append(stellar_distribution, )
    #     sfr = np.append(sfr, H.flatten())


    out_test = tr.transform(np.array([XX.flatten(), YY.flatten()]).T)




    # Plot star formation histories on adjusted coordinates
    # Plot color contours with linear spacing

    if levels is None:
        levels = np.linspace(1.0e-2 * np.max(stellar_distribution), np.max(stellar_distribution), 10)
        sf_plot = plt.tricontourf(out_test[:,0], out_test[:,1], stellar_distribution, cmap=color_map,
                                  extend='max', alpha=contour_alpha, rasterized=True)
    else:
        sf_plot = plt.tricontourf(out_test[:,0], out_test[:,1], stellar_distribution, cmap=color_map,
                                  levels=levels, extend='max', alpha=contour_alpha, rasterized=True)

    if color_bar:
        sf_plot = plt.colorbar()


    if title is not None:
        sf_plot = plt.title(title)



    if circle_ra is not None and circle_dec is not None and circle_radius is not None:

        # Transform distribution
        if (sys.version_info > (3, 0)):
            circle_pos_polar = tr.transform(np.array([circle_ra, circle_dec]).T)  # Python 3 code
        else:
            circle_pos_polar = tr.transform(zip(circle_ra, circle_dec))  # Python 2 code

        for j in range(len(circle_pos_polar[:,0])):
            plt.Circle((circle_pos_polar[j,0], circle_pos_polar[j,1]), 1.0, color='r', fill=False)
            plt.scatter(circle_pos_polar[j,0], circle_pos_polar[j,1], color='r', s=20, marker="*", zorder=10)


    # Plot a star at the coordinate position, if supplied
    if ra is not None and dec is not None:

        # If only a single ra and dec, or a list of points
        if isinstance(ra, np.ndarray):
            coor_pol = tr.transform(np.array([ra, dec]).T)
            sf_plot = plt.scatter(coor_pol[:,0], coor_pol[:,1], color='r', s=25, marker=".", zorder=10)
        else:
            coor_pol1, coor_pol2 = tr.transform(np.array([np.array([ra, ra]), np.array([dec, dec])]).T)
            sf_plot = plt.scatter(coor_pol1[0], coor_pol1[1], color='r', s=50, marker="*", zorder=10)



    return sf_plot, ax1
