import torch

from .granular import MVGBList


def SplitExtract(epoch, rec, args, model, optimizer, data_loader, lossFunList, scheduler, logger):
    device = args.device
    view = args.view
    tot_loss = 0.0
    all_loss = [0.0, 0.0, 0.0, 0.0]
    mes = torch.nn.MSELoss()
    criterion_gra, criterion_LTwo, criterion_Sim = lossFunList[0], lossFunList[1], lossFunList[2]

    if hasattr(criterion_gra, "set_epoch"):
        criterion_gra.set_epoch(epoch)

    for batch_idx, (xs, y, batch_indices) in enumerate(data_loader):
        del batch_indices  
        for v in range(view):
            xs[v] = xs[v].to(device)
        y = y.to(device)

        if rec == "AE":
            optimizer.zero_grad()
            _, lHZs, _, xLNs, xLPss, _, hs, zNoises, zPrivates = model(xs)
        elif rec == "DAE":
            noise_x = []
            for v in range(view):
                noise_x.append(xs[v] + torch.randn_like(xs[v]))
            optimizer.zero_grad()
            _, lHZs, _, xLNs, xLPss, _, hs, zNoises, zPrivates = model(noise_x)
        else:
            raise ValueError(f"Unknown backbone: {rec}")

        loss_list = []
        loss_LTwo_list = []
        loss_far_list = []

        for v in range(view):
            loss_far = criterion_Sim.forward_orthogonal(zNoises[v], zPrivates[v])
            loss_far_list.append(args.lambda1 * loss_far * loss_far)

            ww = 1.0 / view
            for w in range(view):
                loss_list.append(ww * mes(xs[v], xLPss[w][v]))
            loss_LTwo_list.append(args.lambda2 * criterion_LTwo.foward_ZLTwo(zNoises[v]))

        # Module 1: view-specific adaptive granular-ball generation (no k-means).
        mv_gblist = MVGBList(
            hs,
            y,
            p=args.gb_min_samples,
            max_balls=args.gb_max_balls,
            min_gain=args.gb_min_gain,
            min_separation=args.gb_min_separation,
            generation=getattr(args, "gb_generation", "adaptive"),
            kmeans_k=args.class_num,
        )

        # Module 2/3 + uncertainty + anti-drift are all inside MultiviewGCLoss.
        loss_con = criterion_gra(mv_gblist)

        loss_reg = sum(loss_list)
        loss_LTwo = sum(loss_LTwo_list)
        loss_far = sum(loss_far_list)

        loss = loss_reg + loss_con + loss_LTwo + loss_far
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        all_loss[0] += loss_con.item()
        all_loss[1] += loss_reg.item()
        all_loss[2] += loss_LTwo.item()
        all_loss[3] += loss_far.item()

    scheduler.step()
    lossFunList = [criterion_gra, criterion_LTwo, criterion_Sim]

    stats = getattr(criterion_gra, "last_stats", {})
    extra = (
        " | GB(avg)={:.2f}, OT-entropy={:.4f}, drift={:.6f}".format(
            stats.get("gb_count", 0.0),
            stats.get("ot_entropy", 0.0),
            stats.get("drift", 0.0),
        )
        if stats
        else ""
    )
    logger.log(
        "Epoch {}".format(epoch),
        "Loss:{:.6f}. con:{:.6f}, rec:{:.6f}, L2:{:.6f}, NS:{:.6f}{}".format(
            tot_loss / len(data_loader),
            all_loss[0] / len(data_loader),
            all_loss[1] / len(data_loader),
            all_loss[2] / len(data_loader),
            all_loss[3] / len(data_loader),
            extra,
        ),
    )
    return optimizer, lossFunList, scheduler


def ViewsFusion(epoch, rec, args, model, optimizer, data_loader, scheduler, logger):
    device = args.device
    view = args.view
    tot_loss = 0.0
    all_loss = [0.0, 0.0, 0.0]
    mes = torch.nn.MSELoss()

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)

        if rec == "AE":
            optimizer.zero_grad()
            zs, lHZs, hHZs, xLNs, xLPss, xHPs, hs, zNoises, zPrivates = model(xs)
        elif rec == "DAE":
            noise_x = [xs[v] + torch.randn_like(xs[v]) for v in range(view)]
            optimizer.zero_grad()
            zs, lHZs, hHZs, xLNs, xLPss, xHPs, hs, zNoises, zPrivates = model(noise_x)
        else:
            raise ValueError(f"Unknown backbone: {rec}")

        loss_list = [mes(xs[v], xHPs[v]) for v in range(view)]
        loss_tol = sum(loss_list)
        loss_tol.backward()
        optimizer.step()

        tot_loss += loss_tol.item()
        all_loss[0] += loss_tol.item()

    scheduler.step()
    logger.log(
        "Epoch {}".format(epoch),
        "Loss:{:.6f}. loss_tol:{:.6f}".format(
            tot_loss / len(data_loader),
            all_loss[0] / len(data_loader),
        ),
    )
    return optimizer, scheduler
